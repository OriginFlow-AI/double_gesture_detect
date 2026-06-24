#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include "double_ok_gesture/camera.hpp"

namespace {

std::string timestamp_name() {
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return std::to_string(millis) + ".jpg";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        double_ok_gesture::CameraConfig camera;
        std::filesystem::path output_dir = "data/raw/local_validation";
        std::string label;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("Missing value for " + key);
                }
                return argv[++i];
            };
            if (key == "--camera") {
                camera.source = next();
            } else if (key == "--label") {
                label = next();
            } else if (key == "--output-dir") {
                output_dir = next();
            } else if (key == "--width") {
                camera.width = std::stoi(next());
            } else if (key == "--height") {
                camera.height = std::stoi(next());
            } else {
                throw std::invalid_argument("Unknown argument: " + key);
            }
        }
        if (label.empty()) {
            throw std::invalid_argument("--label is required");
        }
        const auto label_dir = output_dir / label;
        std::filesystem::create_directories(label_dir);
        auto stream = double_ok_gesture::open_camera(camera);
        std::cout << "Press SPACE to save a frame; press Q or ESC to quit\n";
        while (true) {
            auto frame = stream.read();
            if (!frame) {
                continue;
            }
            cv::imshow("capture_samples_cpp", *frame);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 'q' || key == 27) {
                break;
            }
            if (key == ' ') {
                const auto path = label_dir / timestamp_name();
                if (!cv::imwrite(path.string(), *frame)) {
                    throw std::runtime_error("Failed to write captured frame: " + path.string());
                }
                std::cout << path << '\n';
            }
        }
        stream.close();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
