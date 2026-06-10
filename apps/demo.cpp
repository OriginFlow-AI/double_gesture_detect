#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include "double_ok_gesture/camera.hpp"
#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/recognizer.hpp"
#include "double_ok_gesture/runtime.hpp"

namespace {

struct Args {
    double_ok_gesture::CameraConfig camera;
    std::filesystem::path config = "configs/default.json";
    std::optional<std::filesystem::path> model;
    std::optional<double> threshold;
    bool capture_gate = false;
    bool require_glasses_pose = false;
    std::optional<std::filesystem::path> glasses_pose;
    bool headless = false;
    double status_interval = 1.0;
    int max_frames = 0;
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
        if (key == "--camera") {
            args.camera.source = next();
        } else if (key == "--config") {
            args.config = next();
        } else if (key == "--model") {
            args.model = next();
        } else if (key == "--threshold") {
            args.threshold = std::stod(next());
        } else if (key == "--width") {
            args.camera.width = std::stoi(next());
        } else if (key == "--height") {
            args.camera.height = std::stoi(next());
        } else if (key == "--camera-fps") {
            args.camera.fps = std::stod(next());
        } else if (key == "--fourcc") {
            args.camera.fourcc = next();
        } else if (key == "--capture-gate") {
            args.capture_gate = true;
        } else if (key == "--require-glasses-pose") {
            args.require_glasses_pose = true;
        } else if (key == "--glasses-pose") {
            args.glasses_pose = next();
        } else if (key == "--headless") {
            args.headless = true;
        } else if (key == "--status-interval") {
            args.status_interval = std::stod(next());
        } else if (key == "--max-frames") {
            args.max_frames = std::stoi(next());
        } else if (key == "--list-cameras") {
            std::cout << double_ok_gesture::format_video_devices() << '\n';
            std::exit(0);
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
        auto runtime_config = double_ok_gesture::load_runtime_config(args.config);
        double_ok_gesture::apply_threshold_override(runtime_config, args.threshold);
        if (args.require_glasses_pose) {
            double_ok_gesture::require_glasses_pose(runtime_config);
        }

        double_ok_gesture::OKHandClassifier classifier = args.model
                                                             ? double_ok_gesture::OKHandClassifier(
                                                                   *args.model,
                                                                   runtime_config.recognizer.ok_threshold)
                                                             : double_ok_gesture::OKHandClassifier(
                                                                   runtime_config.recognizer.ok_threshold);
        double_ok_gesture::DoubleOKRecognizer recognizer(
            classifier,
            static_cast<std::size_t>(runtime_config.recognizer.stable_window),
            static_cast<std::size_t>(runtime_config.recognizer.stable_min_positive));

        auto camera = double_ok_gesture::open_camera(args.camera);
        double_ok_gesture::RuntimeMetrics metrics;
        double last_status = 0.0;
        int frames = 0;
        while (true) {
            auto frame = camera.read();
            if (!frame) {
                continue;
            }
            const double started = double_ok_gesture::monotonic_seconds();
            const double_ok_gesture::DoubleOKResult result = recognizer.process_hands({});
            std::optional<double_ok_gesture::CaptureGateDecision> decision;
            if (args.capture_gate) {
                const auto pose = args.glasses_pose ? double_ok_gesture::load_glasses_pose(*args.glasses_pose) : std::nullopt;
                decision = double_ok_gesture::evaluate_capture_gate(result, runtime_config.capture_gate, pose);
            }
            const auto snapshot = metrics.update(started);
            ++frames;

            const double now = double_ok_gesture::monotonic_seconds();
            if (args.headless || now - last_status >= args.status_interval) {
                std::cout << "hands=" << result.hands.size() << " ok_count=" << result.ok_count
                          << " double_ok=" << result.double_ok << " stable=" << result.stable_double_ok;
                if (decision) {
                    std::cout << " gate_ready=" << decision->ready << " reason="
                              << double_ok_gesture::gate_reason_value(decision->reason);
                }
                std::cout << " fps=" << snapshot.fps << " processing_ms=" << snapshot.processing_ms << '\n';
                last_status = now;
            }
            if (!args.headless) {
                cv::putText(
                    *frame,
                    "C++ build: hand landmark provider not linked",
                    {24, 36},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    {60, 180, 255},
                    2,
                    cv::LINE_AA);
                cv::imshow("double_ok_gesture_cpp", *frame);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 'q' || key == 27) {
                    break;
                }
            }
            if (args.max_frames > 0 && frames >= args.max_frames) {
                break;
            }
        }
        camera.close();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
