#include <QApplication>
#include <iostream>
#include <stdexcept>

#include "double_ok_gesture/demo_app.hpp"
#include "double_ok_gesture/qt_dashboard.hpp"

namespace {

double_ok_gesture::QtDashboardOptions dashboard_options_from_demo_args(const double_ok_gesture::DemoArgs& args) {
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
        QApplication application(argc, argv);
        double_ok_gesture::RuntimeBundle runtime =
            double_ok_gesture::make_runtime(double_ok_gesture::demo_runtime_options(args));
        double_ok_gesture::QtDashboard dashboard(
            application,
            runtime,
            dashboard_options_from_demo_args(args));
        const int exit_code = dashboard.run();
        runtime.camera.close();
        return exit_code;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
