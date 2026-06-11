#include "double_ok_gesture/demo_app.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/capture_writer.hpp"

namespace double_ok_gesture {

DemoArgs parse_demo_args(int argc, char** argv) {
    DemoArgs args;
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
        } else if (key == "--target-fps") {
            args.target_fps = std::stod(next());
        } else if (key == "--dashboard-width") {
            args.dashboard_width = std::stoi(next());
        } else if (key == "--dashboard-height") {
            args.dashboard_height = std::stoi(next());
        } else if (key == "--fullscreen") {
            args.fullscreen = true;
        } else if (key == "--screenshot-dir") {
            args.screenshot_dir = next();
        } else if (key == "--capture-output-dir") {
            args.capture_output_dir = next();
        } else if (key == "--capture-cooldown") {
            args.capture_cooldown_sec = std::stod(next());
        } else if (key == "--disable-auto-capture") {
            args.disable_auto_capture = true;
        } else if (key == "--voice-prompts") {
            // Kept for CLI compatibility. TTS is intentionally not spawned in the C++ build.
        } else if (key == "--prompt-interval") {
            (void)next();
        } else if (key == "--max-frames") {
            args.max_frames = std::stoi(next());
        } else if (key == "--landmark-backend") {
            args.landmark_backend = landmark_backend_from_string(next());
        } else if (key == "--landmarks-json") {
            args.landmarks_json = next();
        } else if (key == "--list-cameras") {
            args.list_cameras = true;
            return args;
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    return args;
}

RuntimeOptions demo_runtime_options(const DemoArgs& args) {
    RuntimeOptions options;
    options.camera = args.camera;
    options.config_path = args.config;
    options.model_path = args.model;
    options.threshold = args.threshold;
    options.require_glasses_pose = args.require_glasses_pose;
    options.capture_output_dir = args.capture_output_dir;
    options.capture_cooldown_sec = args.capture_cooldown_sec;
    options.disable_auto_capture = args.disable_auto_capture;
    options.landmark_backend = args.landmark_backend;
    options.landmarks_json_path = args.landmarks_json;
    return options;
}

ProcessFrameOptions demo_process_frame_options(const DemoArgs& args) {
    ProcessFrameOptions options;
    options.capture_gate = args.capture_gate;
    if (args.glasses_pose) {
        options.glasses_pose = load_glasses_pose(*args.glasses_pose);
    }
    return options;
}

std::optional<std::string> backend_unavailable_message(LandmarkBackend backend) {
    switch (backend) {
        case LandmarkBackend::Rknn:
            return "RKNN hand landmark backend is not available in this build; provide RKNN SDK/model for RV1126.";
        case LandmarkBackend::LandmarksJson:
        case LandmarkBackend::MediaPipe:
        case LandmarkBackend::OpenCVDebug:
        case LandmarkBackend::None:
            return std::nullopt;
    }
    return std::nullopt;
}

int write_demo_camera_list(std::ostream& out) {
    out << format_video_devices() << '\n';
    return 0;
}

int run_demo_headless(const DemoArgs& args, std::ostream& out, std::ostream& err) {
    RuntimeBundle runtime = make_runtime(demo_runtime_options(args));
    CaptureWriter capture_writer(runtime.config.data_capture);
    if (const auto message = backend_unavailable_message(args.landmark_backend)) {
        err << *message << '\n';
    }
    double last_status = 0.0;
    int frames = 0;
    while (true) {
        auto frame = runtime.camera.read();
        if (!frame) {
            continue;
        }
        const auto frame_result = process_runtime_frame(runtime, *frame, demo_process_frame_options(args));
        const auto& result = frame_result.result;
        const auto& decision = frame_result.decision;
        if (decision) {
            if (auto saved = capture_writer.maybe_save(*frame, result, *decision, landmark_backend_value(args.landmark_backend))) {
                out << "capture_saved=" << saved->string() << '\n';
            }
        }
        const auto snapshot = runtime.metrics.update(frame_result.started);
        ++frames;

        const double now = monotonic_seconds();
        if (args.status_interval == 0.0 || now - last_status >= args.status_interval) {
            out << "hands=" << result.hands.size() << " ok_count=" << result.ok_count << " double_ok="
                << result.double_ok << " stable=" << result.stable_double_ok;
            if (decision) {
                out << " gate_ready=" << decision->ready << " reason=" << gate_reason_value(decision->reason);
            }
            out << " fps=" << snapshot.fps << " processing_ms=" << snapshot.processing_ms << '\n';
            last_status = now;
        }
        if (args.max_frames > 0 && frames >= args.max_frames) {
            break;
        }
    }
    runtime.camera.close();
    return 0;
}

}  // namespace double_ok_gesture
