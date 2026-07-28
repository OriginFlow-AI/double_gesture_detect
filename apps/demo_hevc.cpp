#include <QApplication>
#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/demo_app.hpp"
#include "double_ok_gesture/hevc_async_reader.hpp"
#include "double_ok_gesture/qt_dashboard.hpp"

class SafeQtApplication : public QApplication {
public:
    using QApplication::QApplication;

    bool notify(QObject* receiver, QEvent* event) override {
        try {
            return QApplication::notify(receiver, event);
        } catch (const std::exception& e) {
            std::cerr << "Qt exception in " << (receiver ? receiver->objectName().toStdString() : "null")
                      << " event type " << event->type() << ": " << e.what() << std::endl;
            return false;
        }
    }
};

namespace {

double_ok_gesture::QtDashboardOptions dashboard_options_from_demo_args(
    const double_ok_gesture::DemoArgs& args) {
    double_ok_gesture::QtDashboardOptions options;
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
    if (args.right_half) {
        options.crop_x = 320;
        options.crop_y = 180;
        options.crop_width = 1280;
        options.crop_height = 720;
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = double_ok_gesture::parse_demo_args(argc, argv);

        SafeQtApplication application(argc, argv);

        double_ok_gesture::RuntimeOptions runtime_opts =
            double_ok_gesture::demo_runtime_options(args);
        double_ok_gesture::configure_logging(runtime_opts.log_level);

        auto runtime_config = double_ok_gesture::load_runtime_config(runtime_opts.config_path);
        double_ok_gesture::apply_threshold_override(runtime_config, runtime_opts.threshold);
        if (runtime_opts.require_glasses_pose) {
            double_ok_gesture::require_glasses_pose(runtime_config);
        }
        if (runtime_opts.capture_output_dir) {
            runtime_config.data_capture.output_dir = *runtime_opts.capture_output_dir;
        }
        if (runtime_opts.capture_cooldown_sec) {
            runtime_config.data_capture.cooldown_sec = *runtime_opts.capture_cooldown_sec;
        }
        if (runtime_opts.disable_auto_capture) {
            runtime_config.data_capture.enabled = false;
        }
        double_ok_gesture::validate_runtime_config(runtime_config);

        double_ok_gesture::OKHandClassifier classifier(runtime_config.recognizer.ok_threshold);
        double_ok_gesture::DoubleOKRecognizer recognizer(
            classifier,
            static_cast<std::size_t>(runtime_config.recognizer.stable_window),
            static_cast<std::size_t>(runtime_config.recognizer.stable_min_positive));

        auto landmark_provider = double_ok_gesture::make_landmark_provider(
            runtime_opts, runtime_config);

        // 判断输入源类型：/dev/videoX 走 v4l2src，其他路径走 filesrc
        const bool is_file_input = runtime_opts.camera.source.find("/dev/video") != 0;
        auto hevc_reader = std::make_unique<double_ok_gesture::HevcAsyncReader>(
            runtime_opts.camera.source,
            runtime_opts.camera.width,
            runtime_opts.camera.height,
            runtime_opts.camera.fps,
            2,
            is_file_input
                ? double_ok_gesture::HevcAsyncReader::SourceType::FILE
                : double_ok_gesture::HevcAsyncReader::SourceType::V4L2,
            args.loop_file);

        double_ok_gesture::log_message(
            double_ok_gesture::LogLevel::Info,
            std::string("runtime initialized (HEVC async): backend=") +
                double_ok_gesture::landmark_backend_value(runtime_opts.landmark_backend) +
                ", config=" + runtime_opts.config_path.string());

        double_ok_gesture::RuntimeBundle runtime;
        runtime.config = std::move(runtime_config);
        runtime.classifier = std::move(classifier);
        runtime.recognizer = std::move(recognizer);
        runtime.landmark_provider = std::move(landmark_provider);
        runtime.camera = double_ok_gesture::CameraStream{};
        runtime.metrics = double_ok_gesture::RuntimeMetrics();
        runtime.landmark_backend = runtime_opts.landmark_backend;
        runtime.right_half = runtime_opts.right_half;

        double_ok_gesture::QtDashboard dashboard(
            application,
            runtime,
            dashboard_options_from_demo_args(args));

        const int exit_code = dashboard.run_hevc(std::move(hevc_reader));
        return exit_code;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
