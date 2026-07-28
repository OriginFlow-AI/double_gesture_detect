#include "double_ok_gesture/demo_app.hpp"

#include <stdexcept>
#include <string>

#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/cli.hpp"
#include "double_ok_gesture/qt_dashboard.hpp"

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
        } else if (key == "--palm-model") {
            args.palm_model = next();
        } else if (key == "--hand-model" || key == "--pose-model") {
            args.hand_model = next();
        } else if (key == "--threshold") {
            args.threshold = parse_finite_double_argument(next(), key);
        } else if (key == "--width") {
            args.camera.width = parse_int_argument(next(), key);
        } else if (key == "--height") {
            args.camera.height = parse_int_argument(next(), key);
        } else if (key == "--camera-fps") {
            args.camera.fps = parse_finite_double_argument(next(), key);
        } else if (key == "--fourcc") {
            args.camera.fourcc = next();
        } else if (key == "--capture-gate") {
            args.capture_gate = true;
        } else if (key == "--require-glasses-pose") {
            args.require_glasses_pose = true;
        } else if (key == "--glasses-pose") {
            args.glasses_pose = next();
        } else if (key == "--target-fps") {
            args.target_fps = parse_finite_double_argument(next(), key);
        } else if (key == "--dashboard-width") {
            args.dashboard_width = parse_int_argument(next(), key);
        } else if (key == "--dashboard-height") {
            args.dashboard_height = parse_int_argument(next(), key);
        } else if (key == "--fullscreen") {
            args.fullscreen = true;
        } else if (key == "--screenshot-dir") {
            args.screenshot_dir = next();
        } else if (key == "--capture-output-dir") {
            args.capture_output_dir = next();
        } else if (key == "--capture-cooldown") {
            args.capture_cooldown_sec =
                parse_finite_double_argument(next(), key);
        } else if (key == "--disable-auto-capture") {
            args.disable_auto_capture = true;
        } else if (key == "--log-level") {
            args.log_level = next();
        } else if (key == "--max-frames") {
            args.max_frames = parse_int_argument(next(), key);
        } else if (key == "--detection-interval") {
            // 跳帧检测：每 N 帧做一次推理（0/1 = 每帧检测，10 = 每 10 帧检测一次）
            args.detection_skip_frames = parse_int_argument(next(), key);
        } else if (key == "--loop-file") {
            // 文件输入时循环播放（EOS 时 seek 回 0）
            args.loop_file = true;
        } else if (key == "--landmark-backend") {
            args.landmark_backend = landmark_backend_from_string(next());
        } else if (key == "--landmarks-json") {
            args.landmarks_json = next();
        } else if (key == "--right-half") {
            args.right_half = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    if (args.target_fps < 0.0) {
        throw std::invalid_argument("--target-fps must be non-negative");
    }
    if (args.dashboard_width < 1 || args.dashboard_height < 1) {
        throw std::invalid_argument(
            "--dashboard-width and --dashboard-height must be positive");
    }
    if (args.max_frames < 0) {
        throw std::invalid_argument("--max-frames must be non-negative");
    }
    return args;
}

RuntimeOptions demo_runtime_options(const DemoArgs& args) {
    RuntimeOptions options;
    options.camera = args.camera;
    options.config_path = args.config;
    options.palm_model_path = args.palm_model;
    options.hand_model_path = args.hand_model;
    options.threshold = args.threshold;
    options.require_glasses_pose = args.require_glasses_pose;
    options.capture_output_dir = args.capture_output_dir;
    options.capture_cooldown_sec = args.capture_cooldown_sec;
    options.disable_auto_capture = args.disable_auto_capture;
    options.log_level = args.log_level;
    options.landmark_backend = args.landmark_backend;
    options.landmarks_json_path = args.landmarks_json;
    options.right_half = args.right_half;
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

QtDashboardOptions demo_dashboard_options(const DemoArgs& args) {
    QtDashboardOptions options;
    options.width = args.dashboard_width;
    options.height = args.dashboard_height;
    options.fullscreen = args.fullscreen;
    options.screenshot_dir = args.screenshot_dir;
    options.max_frames = args.max_frames;
    options.target_fps = args.target_fps;
    options.capture_gate = args.capture_gate;
    options.glasses_pose = args.glasses_pose;
    options.landmark_backend = args.landmark_backend;
    options.palm_model_path = args.palm_model;
    options.hand_model_path = args.hand_model;
    options.right_half = args.right_half;
    options.detection_skip_frames = args.detection_skip_frames;
    return options;
}

}  // namespace double_ok_gesture
