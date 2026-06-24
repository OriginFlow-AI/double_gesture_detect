#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/camera.hpp"

int main(int argc, char** argv) {
    try {
        double_ok_gesture::CameraConfig settings;
        bool probe = false;
        bool list = argc == 1;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("Missing value for " + key);
                }
                return argv[++i];
            };
            if (key == "--camera") {
                settings.source = next();
            } else if (key == "--list") {
                list = true;
            } else if (key == "--probe") {
                probe = true;
            } else if (key == "--width") {
                settings.width = std::stoi(next());
            } else if (key == "--height") {
                settings.height = std::stoi(next());
            } else if (key == "--fps") {
                settings.fps = std::stod(next());
            } else if (key == "--fourcc") {
                settings.fourcc = next();
            } else {
                throw std::invalid_argument("Unknown argument: " + key);
            }
        }
        if (list) {
            std::cout << double_ok_gesture::format_video_devices() << '\n';
        }
        if (probe) {
            auto stream = double_ok_gesture::open_camera(settings);
            auto frame = stream.read();
            if (!frame) {
                throw std::runtime_error("Camera probe did not receive a frame");
            }
            const auto& info = stream.info();
            std::cout << "OK " << info.source << ' ' << info.width << 'x' << info.height << ' ' << info.fps << "FPS "
                      << info.fourcc << '\n';
            stream.close();
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
