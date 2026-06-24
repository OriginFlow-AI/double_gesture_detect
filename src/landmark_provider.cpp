#include "double_ok_gesture/landmark_provider.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include "double_ok_gesture/features.hpp"
#include "double_ok_gesture/json.hpp"

namespace double_ok_gesture {
namespace {

Point3 parse_point(const Json& value) {
    if (value.is_array()) {
        const auto& coords = value.as_array();
        if (coords.size() < 2 || coords.size() > 3) {
            throw std::runtime_error("landmark point array must contain [x, y] or [x, y, z]");
        }
        return {
            coords[0].as_number(),
            coords[1].as_number(),
            coords.size() >= 3 ? coords[2].as_number() : 0.0,
        };
    }
    if (value.is_object()) {
        const auto* x = value.get("x");
        const auto* y = value.get("y");
        const auto* z = value.get("z");
        if (!x || !y) {
            throw std::runtime_error("landmark point object must contain x and y");
        }
        return {
            x->as_number(),
            y->as_number(),
            z ? z->as_number() : 0.0,
        };
    }
    throw std::runtime_error("landmark point must be an array or object");
}

Landmarks parse_landmarks(const Json& value) {
    const auto& points = value.as_array();
    if (points.size() != 21) {
        throw std::runtime_error("landmarks-json hand must contain exactly 21 points");
    }
    Landmarks landmarks{};
    for (std::size_t index = 0; index < landmarks.size(); ++index) {
        landmarks[index] = parse_point(points[index]);
    }
    validate_landmarks(landmarks);
    return landmarks;
}

DetectedHand parse_hand(const Json& value, std::size_t index) {
    if (value.is_array()) {
        return {
            parse_landmarks(value),
            index == 0 ? "Left" : (index == 1 ? "Right" : "Unknown"),
            std::nullopt,
            false,
        };
    }
    if (!value.is_object()) {
        throw std::runtime_error("landmarks-json hand must be an object or 21-point array");
    }
    const auto* landmarks = value.get("landmarks");
    if (!landmarks) {
        landmarks = value.get("hand_landmarks");
    }
    if (!landmarks) {
        throw std::runtime_error("landmarks-json hand object must contain landmarks");
    }
    const auto* handedness = value.get("handedness");
    const auto* score = value.get("ok_score");
    if (!score) {
        score = value.get("score");
    }
    return {
        parse_landmarks(*landmarks),
        handedness ? handedness->as_string() : (index == 0 ? "Left" : (index == 1 ? "Right" : "Unknown")),
        score ? std::optional<double>(score->as_number()) : std::nullopt,
        false,
    };
}

std::vector<DetectedHand> parse_hands_json(const Json& root) {
    const Json* hands_value = &root;
    if (root.is_object()) {
        hands_value = root.get("hands");
        if (!hands_value) {
            throw std::runtime_error("landmarks-json root object must contain hands");
        }
    }
    const auto& hands = hands_value->as_array();
    std::vector<DetectedHand> detected;
    detected.reserve(hands.size());
    for (std::size_t index = 0; index < hands.size(); ++index) {
        detected.push_back(parse_hand(hands[index], index));
    }
    return detected;
}

std::filesystem::path default_mediapipe_python_path() {
    if (const char* value = std::getenv("DOUBLE_OK_MEDIAPIPE_PYTHON"); value && value[0] != '\0') {
        return value;
    }
    if (std::filesystem::exists(".venv/bin/python")) {
        return ".venv/bin/python";
    }
    if (std::filesystem::exists("../.venv/bin/python")) {
        return "../.venv/bin/python";
    }
    return "python3";
}

std::filesystem::path default_mediapipe_script_path() {
    if (const char* value = std::getenv("DOUBLE_OK_MEDIAPIPE_SERVER"); value && value[0] != '\0') {
        return value;
    }
    if (std::filesystem::exists("scripts/mediapipe_landmark_server.py")) {
        return "scripts/mediapipe_landmark_server.py";
    }
    return "../scripts/mediapipe_landmark_server.py";
}

[[noreturn]] void throw_errno(const std::string& action) {
    throw std::runtime_error(action + ": " + std::strerror(errno));
}

void write_all(int fd, const unsigned char* data, std::size_t size) {
    while (size > 0) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw_errno("Cannot write to MediaPipe landmark sidecar");
        }
        if (written == 0) {
            throw std::runtime_error("MediaPipe landmark sidecar pipe closed while writing");
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

std::string read_line(int fd) {
    std::string line;
    char ch = '\0';
    while (true) {
        const ssize_t count = ::read(fd, &ch, 1);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw_errno("Cannot read from MediaPipe landmark sidecar");
        }
        if (count == 0) {
            throw std::runtime_error("MediaPipe landmark sidecar exited before sending a complete response");
        }
        if (ch == '\n') {
            return line;
        }
        line.push_back(ch);
    }
}

bool ready_response(const std::string& line) {
    const Json root = parse_json(line);
    const Json* ready = root.get("ready");
    return ready && ready->as_bool();
}

}  // namespace

NullHandLandmarkProvider::NullHandLandmarkProvider(std::string name, bool available)
    : info_{std::move(name), available, false} {}

LandmarkProviderInfo NullHandLandmarkProvider::info() const {
    return info_;
}

std::vector<DetectedHand> NullHandLandmarkProvider::detect(const cv::Mat&) {
    return {};
}

OpenCVDebugLandmarkProvider::OpenCVDebugLandmarkProvider(HandDetectorConfig config) : detector_(config) {}

LandmarkProviderInfo OpenCVDebugLandmarkProvider::info() const {
    return {"opencv-heuristic", true, true};
}

std::vector<DetectedHand> OpenCVDebugLandmarkProvider::detect(const cv::Mat& frame_bgr) {
    return detector_.detect(frame_bgr);
}

JsonHandLandmarkProvider::JsonHandLandmarkProvider(std::filesystem::path path) : path_(std::move(path)) {}

LandmarkProviderInfo JsonHandLandmarkProvider::info() const {
    return {"landmarks-json", true, false};
}

std::vector<DetectedHand> JsonHandLandmarkProvider::detect(const cv::Mat&) {
    return parse_hands_json(load_json(path_));
}

struct MediaPipePythonLandmarkProvider::Impl {
    Impl(std::filesystem::path python_path, std::filesystem::path script_path)
        : python_path(std::move(python_path)), script_path(std::move(script_path)) {
        start();
    }

    ~Impl() {
        stop();
    }

    void stop() {
        if (stdin_fd >= 0) {
            ::close(stdin_fd);
            stdin_fd = -1;
        }
        if (stdout_fd >= 0) {
            ::close(stdout_fd);
            stdout_fd = -1;
        }
        if (pid > 0) {
            int status = 0;
            if (::waitpid(pid, &status, WNOHANG) == 0) {
                ::kill(pid, SIGTERM);
                (void)::waitpid(pid, &status, 0);
            }
            pid = -1;
        }
    }

    void start() {
        if (!std::filesystem::exists(script_path)) {
            throw std::runtime_error("MediaPipe landmark sidecar script not found: " + script_path.string());
        }
        std::signal(SIGPIPE, SIG_IGN);
        std::array<int, 2> to_child{-1, -1};
        std::array<int, 2> from_child{-1, -1};
        if (::pipe(to_child.data()) != 0) {
            throw_errno("Cannot create MediaPipe sidecar input pipe");
        }
        if (::pipe(from_child.data()) != 0) {
            const int saved_errno = errno;
            ::close(to_child[0]);
            ::close(to_child[1]);
            errno = saved_errno;
            throw_errno("Cannot create MediaPipe sidecar output pipe");
        }

        pid = ::fork();
        if (pid < 0) {
            const int saved_errno = errno;
            ::close(to_child[0]);
            ::close(to_child[1]);
            ::close(from_child[0]);
            ::close(from_child[1]);
            errno = saved_errno;
            throw_errno("Cannot fork MediaPipe landmark sidecar");
        }
        if (pid == 0) {
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            ::close(to_child[0]);
            ::close(to_child[1]);
            ::close(from_child[0]);
            ::close(from_child[1]);
            const std::string python = python_path.string();
            const std::string script = script_path.string();
            ::execlp(python.c_str(), python.c_str(), script.c_str(), static_cast<char*>(nullptr));
            std::perror("exec mediapipe landmark sidecar");
            ::_exit(127);
        }

        ::close(to_child[0]);
        ::close(from_child[1]);
        stdin_fd = to_child[1];
        stdout_fd = from_child[0];

        try {
            const std::string line = read_line(stdout_fd);
            if (!ready_response(line)) {
                throw std::runtime_error("MediaPipe landmark sidecar did not send ready response");
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) {
        if (frame_bgr.empty()) {
            return {};
        }
        std::vector<unsigned char> encoded;
        const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 85};
        if (!cv::imencode(".jpg", frame_bgr, encoded, params)) {
            throw std::runtime_error("Cannot encode frame for MediaPipe landmark sidecar");
        }
        if (encoded.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Encoded frame is too large for MediaPipe landmark sidecar protocol");
        }
        const std::uint32_t length = static_cast<std::uint32_t>(encoded.size());
        const std::array<unsigned char, 4> header{
            static_cast<unsigned char>(length & 0xffU),
            static_cast<unsigned char>((length >> 8U) & 0xffU),
            static_cast<unsigned char>((length >> 16U) & 0xffU),
            static_cast<unsigned char>((length >> 24U) & 0xffU),
        };
        write_all(stdin_fd, header.data(), header.size());
        write_all(stdin_fd, encoded.data(), encoded.size());
        return parse_hands_json(parse_json(read_line(stdout_fd)));
    }

    std::filesystem::path python_path;
    std::filesystem::path script_path;
    pid_t pid = -1;
    int stdin_fd = -1;
    int stdout_fd = -1;
};

MediaPipePythonLandmarkProvider::MediaPipePythonLandmarkProvider()
    : MediaPipePythonLandmarkProvider(default_mediapipe_python_path(), default_mediapipe_script_path()) {}

MediaPipePythonLandmarkProvider::MediaPipePythonLandmarkProvider(
    std::filesystem::path python_path,
    std::filesystem::path script_path)
    : impl_(std::make_unique<Impl>(std::move(python_path), std::move(script_path))) {}

MediaPipePythonLandmarkProvider::~MediaPipePythonLandmarkProvider() = default;

LandmarkProviderInfo MediaPipePythonLandmarkProvider::info() const {
    return {"mediapipe-python", true, false};
}

std::vector<DetectedHand> MediaPipePythonLandmarkProvider::detect(const cv::Mat& frame_bgr) {
    return impl_->detect(frame_bgr);
}

}  // namespace double_ok_gesture
