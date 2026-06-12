#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/capture_writer.hpp"
#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/demo_app.hpp"
#include "double_ok_gesture/features.hpp"
#include "double_ok_gesture/hand_detector.hpp"
#include "double_ok_gesture/json.hpp"
#include "double_ok_gesture/landmark_provider.hpp"
#include "double_ok_gesture/live_ui.hpp"
#include "double_ok_gesture/model_io.hpp"
#include "double_ok_gesture/recognizer.hpp"
#include "double_ok_gesture/report.hpp"
#include "double_ok_gesture/runtime.hpp"
#include "double_ok_gesture/runtime_pipeline.hpp"
#include "double_ok_gesture/training.hpp"

namespace {

#define EXPECT_TRUE(expr)                                                                                              \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            throw std::runtime_error(std::string("EXPECT_TRUE failed: ") + #expr);                                    \
        }                                                                                                              \
    } while (false)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ(lhs, rhs)                                                                                            \
    do {                                                                                                               \
        const auto lhs_value = (lhs);                                                                                  \
        const auto rhs_value = (rhs);                                                                                  \
        if (!(lhs_value == rhs_value)) {                                                                               \
            throw std::runtime_error(std::string("EXPECT_EQ failed: ") + #lhs + " != " + #rhs);                      \
        }                                                                                                              \
    } while (false)

#define EXPECT_NEAR(lhs, rhs, eps)                                                                                     \
    do {                                                                                                               \
        if (std::abs((lhs) - (rhs)) > (eps)) {                                                                         \
            throw std::runtime_error(std::string("EXPECT_NEAR failed: ") + #lhs + " != " + #rhs);                    \
        }                                                                                                              \
    } while (false)

double_ok_gesture::Landmarks make_ok_landmarks() {
    double_ok_gesture::Landmarks pts{};
    pts[0] = {0.0, 0.0, 0.0};
    pts[1] = {0.35, -0.35, 0.0};
    pts[2] = {0.28, -0.52, 0.0};
    pts[3] = {0.18, -0.62, 0.0};
    pts[4] = {0.08, -0.60, 0.0};
    pts[5] = {-0.35, -0.75, 0.0};
    pts[6] = {-0.20, -0.65, 0.0};
    pts[7] = {-0.05, -0.60, 0.0};
    pts[8] = {0.08, -0.60, 0.0};
    pts[9] = {-0.05, -1.00, 0.0};
    pts[10] = {-0.08, -1.65, 0.0};
    pts[11] = {-0.10, -2.25, 0.0};
    pts[12] = {-0.12, -2.90, 0.0};
    pts[13] = {0.22, -0.92, 0.0};
    pts[14] = {0.30, -1.50, 0.0};
    pts[15] = {0.36, -2.02, 0.0};
    pts[16] = {0.42, -2.55, 0.0};
    pts[17] = {0.48, -0.80, 0.0};
    pts[18] = {0.60, -1.22, 0.0};
    pts[19] = {0.70, -1.58, 0.0};
    pts[20] = {0.80, -1.95, 0.0};
    return pts;
}

double_ok_gesture::Landmarks make_open_palm_landmarks() {
    auto pts = make_ok_landmarks();
    pts[4] = {0.75, -0.85, 0.0};
    pts[8] = {-0.38, -2.45, 0.0};
    return pts;
}

double_ok_gesture::HandPrediction make_hand(double center_x, double center_y, bool is_ok = true) {
    double_ok_gesture::Landmarks pts{};
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const double offset = -0.02 + 0.04 * static_cast<double>(i) / static_cast<double>(pts.size() - 1);
        pts[i] = {center_x + offset, center_y, 0.0};
    }
    return {"Left", is_ok ? 0.9 : 0.1, is_ok, pts};
}

double_ok_gesture::DoubleOKResult make_result(std::vector<double_ok_gesture::HandPrediction> hands, bool stable = true) {
    int ok_count = 0;
    for (const auto& hand : hands) {
        ok_count += hand.is_ok ? 1 : 0;
    }
    return {std::move(hands), ok_count >= 2, stable, ok_count};
}

double_ok_gesture::DemoArgs parse_demo_args_for_test(std::vector<std::string> values) {
    std::vector<char*> argv;
    argv.reserve(values.size());
    for (auto& value : values) {
        argv.push_back(value.data());
    }
    return double_ok_gesture::parse_demo_args(static_cast<int>(argv.size()), argv.data());
}

void test_feature_vector_has_stable_shape() {
    const auto vector = double_ok_gesture::feature_vector(make_ok_landmarks(), "Left");
    EXPECT_EQ(vector.size(), 96U);
    for (double value : vector) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

void test_rule_score_prefers_ok_over_open_palm() {
    const double ok_score = double_ok_gesture::rule_ok_score(make_ok_landmarks(), "Left");
    const double palm_score = double_ok_gesture::rule_ok_score(make_open_palm_landmarks(), "Left");
    EXPECT_TRUE(ok_score > 0.65);
    EXPECT_TRUE(palm_score < ok_score);
}

void test_capture_gate_ready_and_blocks() {
    const auto ready = make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5)});
    const auto decision = double_ok_gesture::evaluate_capture_gate(
        ready,
        double_ok_gesture::CaptureGateConfig{.require_glasses_pose = true},
        double_ok_gesture::GlassesPose{0.0, 0.0, 0.0});
    EXPECT_TRUE(decision.ready);
    EXPECT_EQ(decision.reason, double_ok_gesture::GateReason::Ready);

    const auto missing_pose = double_ok_gesture::evaluate_capture_gate(
        ready,
        double_ok_gesture::CaptureGateConfig{.require_glasses_pose = true});
    EXPECT_FALSE(missing_pose.ready);
    EXPECT_EQ(missing_pose.reason, double_ok_gesture::GateReason::GlassesPoseMissing);

    const auto not_centered = double_ok_gesture::evaluate_capture_gate(make_result({make_hand(0.08, 0.5), make_hand(0.6, 0.5)}));
    EXPECT_EQ(not_centered.reason, double_ok_gesture::GateReason::HandsNotCentered);
}

void test_capture_gate_requires_double_ok_and_centered() {
    const auto double_ok_not_centered =
        double_ok_gesture::evaluate_capture_gate(make_result({make_hand(0.08, 0.5), make_hand(0.6, 0.5)}));
    EXPECT_FALSE(double_ok_not_centered.ready);
    EXPECT_TRUE(double_ok_not_centered.double_ok);
    EXPECT_FALSE(double_ok_not_centered.hands_centered);
    EXPECT_EQ(double_ok_not_centered.reason, double_ok_gesture::GateReason::HandsNotCentered);

    const auto centered_not_double_ok =
        double_ok_gesture::evaluate_capture_gate(make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5, false)}));
    EXPECT_FALSE(centered_not_double_ok.ready);
    EXPECT_TRUE(centered_not_double_ok.hands_centered);
    EXPECT_FALSE(centered_not_double_ok.double_ok);
    EXPECT_EQ(centered_not_double_ok.reason, double_ok_gesture::GateReason::NeedDoubleOK);

    const auto centered_double_ok =
        double_ok_gesture::evaluate_capture_gate(make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5)}));
    EXPECT_TRUE(centered_double_ok.ready);
    EXPECT_TRUE(centered_double_ok.double_ok);
    EXPECT_TRUE(centered_double_ok.hands_centered);
    EXPECT_EQ(centered_double_ok.reason, double_ok_gesture::GateReason::Ready);
}

void test_capture_gate_requires_stable_double_ok() {
    const auto unstable_double_ok =
        double_ok_gesture::evaluate_capture_gate(make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5)}, false));
    EXPECT_FALSE(unstable_double_ok.ready);
    EXPECT_FALSE(unstable_double_ok.double_ok);
    EXPECT_EQ(unstable_double_ok.reason, double_ok_gesture::GateReason::NeedDoubleOK);
}

void test_negative_capture_gate() {
    const auto negative = make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5, false)});
    const auto decision = double_ok_gesture::evaluate_labeled_capture_gate(negative, "not_double_ok");
    EXPECT_TRUE(decision.ready);
    EXPECT_TRUE(decision.gesture_ok);

    const auto positive = make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5)});
    const auto rejected = double_ok_gesture::evaluate_labeled_capture_gate(positive, "not_double_ok");
    EXPECT_EQ(rejected.reason, double_ok_gesture::GateReason::AvoidDoubleOK);
}

void test_recognizer_stability() {
    double_ok_gesture::DoubleOKRecognizer recognizer(double_ok_gesture::OKHandClassifier(0.5), 3, 2);
    const double_ok_gesture::DetectedHand left{make_ok_landmarks(), "Left", std::nullopt, false};
    const double_ok_gesture::DetectedHand right{make_ok_landmarks(), "Right", std::nullopt, false};
    const auto first = recognizer.process_hands({left, right});
    const auto second = recognizer.process_hands({left, right});
    EXPECT_TRUE(first.double_ok);
    EXPECT_FALSE(first.stable_double_ok);
    EXPECT_TRUE(second.stable_double_ok);
}

void test_opencv_hand_detector_finds_skin_colored_regions() {
    cv::Mat image(360, 640, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::ellipse(image, {200, 180}, {70, 105}, 0, 0, 360, cv::Scalar(80, 130, 200), -1, cv::LINE_AA);
    cv::ellipse(image, {440, 180}, {70, 105}, 0, 0, 360, cv::Scalar(80, 130, 200), -1, cv::LINE_AA);
    double_ok_gesture::OpenCVHandDetector detector({2, 0.003, 3.0});

    const auto hands = detector.detect(image);

    EXPECT_EQ(hands.size(), 2U);
    EXPECT_TRUE(hands[0].ok_score.has_value());
    EXPECT_TRUE(hands[0].landmarks_estimated);
    EXPECT_TRUE(*hands[0].ok_score >= 0.0);
    EXPECT_TRUE(*hands[0].ok_score <= 1.0);
}

void test_estimated_hands_do_not_draw_fake_keypoint_skeleton() {
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    auto hand = make_hand(0.5, 0.5);
    hand.landmarks_estimated = true;

    double_ok_gesture::draw_hand_tracking(image, double_ok_gesture::DoubleOKResult{{hand}, false, false, 0});

    const cv::Vec3b center = image.at<cv::Vec3b>(100, 100);
    EXPECT_EQ(static_cast<int>(center[0]), 0);
    EXPECT_EQ(static_cast<int>(center[1]), 0);
    EXPECT_EQ(static_cast<int>(center[2]), 0);
}

void expect_dashboard_content(const cv::Mat& dashboard, int width, int height) {
    EXPECT_FALSE(dashboard.empty());
    EXPECT_EQ(dashboard.cols, width);
    EXPECT_EQ(dashboard.rows, height);
    EXPECT_EQ(dashboard.type(), CV_8UC3);

    cv::Mat gray;
    cv::cvtColor(dashboard, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray, mean, stddev);
    EXPECT_TRUE(mean[0] > 10.0);
    EXPECT_TRUE(stddev[0] > 5.0);
}

void test_dashboard_renders_compact_and_regular_layouts() {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(38, 44, 52));
    cv::rectangle(frame, {80, 80}, {560, 400}, cv::Scalar(68, 86, 112), -1, cv::LINE_AA);
    cv::line(frame, {0, 240}, {640, 240}, cv::Scalar(120, 130, 145), 2, cv::LINE_AA);

    const auto result = make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5)});
    const auto decision = double_ok_gesture::evaluate_capture_gate(result);
    const double_ok_gesture::RuntimeSnapshot snapshot{28.4, 11.7, 42};

    const auto compact = double_ok_gesture::render_dashboard(
        frame,
        result,
        decision,
        snapshot,
        "/dev/video2 640x480 30 FPS MJPG",
        25.0,
        "geometry rules",
        960,
        600);
    expect_dashboard_content(compact, 960, 600);

    const auto regular = double_ok_gesture::render_dashboard(
        frame,
        result,
        decision,
        snapshot,
        "/dev/video2 640x480 30 FPS MJPG",
        25.0,
        "geometry rules",
        1440,
        810);
    expect_dashboard_content(regular, 1440, 810);

    const auto waiting = double_ok_gesture::render_dashboard(
        cv::Mat{},
        double_ok_gesture::DoubleOKResult{},
        std::nullopt,
        double_ok_gesture::RuntimeSnapshot{},
        "/dev/video2 640x480 30 FPS MJPG",
        25.0,
        "geometry rules",
        960,
        600);
    expect_dashboard_content(waiting, 960, 600);
}

void test_runtime_metrics() {
    double_ok_gesture::RuntimeMetrics metrics(3);
    metrics.update(0.0, 0.05);
    const auto second = metrics.update(0.5, 0.55);
    const auto third = metrics.update(1.0, 1.05);
    EXPECT_NEAR(second.fps, 2.0, 1e-9);
    EXPECT_NEAR(third.fps, 2.0, 1e-9);
    EXPECT_NEAR(third.processing_ms, 50.0, 1e-9);
}

void test_landmark_backend_parser() {
    EXPECT_EQ(double_ok_gesture::landmark_backend_from_string("rknn"), double_ok_gesture::LandmarkBackend::Rknn);
    EXPECT_EQ(double_ok_gesture::landmark_backend_from_string("mediapipe"), double_ok_gesture::LandmarkBackend::MediaPipe);
    EXPECT_EQ(
        double_ok_gesture::landmark_backend_from_string("landmarks-json"),
        double_ok_gesture::LandmarkBackend::LandmarksJson);
    EXPECT_EQ(
        double_ok_gesture::landmark_backend_from_string("opencv-heuristic"),
        double_ok_gesture::LandmarkBackend::OpenCVDebug);
    EXPECT_EQ(double_ok_gesture::landmark_backend_from_string("none"), double_ok_gesture::LandmarkBackend::None);
    EXPECT_TRUE(double_ok_gesture::landmark_backend_available_in_current_build(double_ok_gesture::LandmarkBackend::LandmarksJson));
    EXPECT_TRUE(double_ok_gesture::landmark_backend_available_in_current_build(double_ok_gesture::LandmarkBackend::MediaPipe));
    EXPECT_TRUE(double_ok_gesture::landmark_backend_available_in_current_build(double_ok_gesture::LandmarkBackend::OpenCVDebug));
    EXPECT_FALSE(double_ok_gesture::landmark_backend_available_in_current_build(double_ok_gesture::LandmarkBackend::Rknn));

    bool threw = false;
    try {
        (void)double_ok_gesture::landmark_backend_from_string("debug");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

void test_demo_args_parse_and_map_options() {
    const auto args = parse_demo_args_for_test({
        "double-ok-demo",
        "--camera",
        "/dev/video42",
        "--config",
        "configs/test.json",
        "--model",
        "models/test.txt",
        "--threshold",
        "0.75",
        "--width",
        "800",
        "--height",
        "600",
        "--camera-fps",
        "30",
        "--fourcc",
        "MJPG",
        "--capture-gate",
        "--require-glasses-pose",
        "--glasses-pose",
        "/tmp/pose.json",
        "--headless",
        "--status-interval",
        "0",
        "--target-fps",
        "15",
        "--dashboard-width",
        "1200",
        "--dashboard-height",
        "720",
        "--fullscreen",
        "--screenshot-dir",
        "reports/test_live",
        "--capture-output-dir",
        "data/raw/test_capture",
        "--capture-cooldown",
        "2.5",
        "--disable-auto-capture",
        "--voice-prompts",
        "--prompt-interval",
        "5",
        "--max-frames",
        "3",
        "--landmark-backend",
        "opencv-heuristic",
        "--landmarks-json",
        "configs/debug_landmarks.json",
    });

    EXPECT_EQ(args.camera.source, std::string("/dev/video42"));
    EXPECT_EQ(args.camera.width, 800);
    EXPECT_EQ(args.camera.height, 600);
    EXPECT_NEAR(args.camera.fps, 30.0, 1e-12);
    EXPECT_EQ(args.camera.fourcc, std::string("MJPG"));
    EXPECT_EQ(args.config.string(), std::string("configs/test.json"));
    EXPECT_TRUE(args.model.has_value());
    EXPECT_EQ(args.model->string(), std::string("models/test.txt"));
    EXPECT_TRUE(args.threshold.has_value());
    EXPECT_NEAR(*args.threshold, 0.75, 1e-12);
    EXPECT_TRUE(args.capture_gate);
    EXPECT_TRUE(args.require_glasses_pose);
    EXPECT_TRUE(args.glasses_pose.has_value());
    EXPECT_TRUE(args.headless);
    EXPECT_NEAR(args.status_interval, 0.0, 1e-12);
    EXPECT_NEAR(args.target_fps, 15.0, 1e-12);
    EXPECT_EQ(args.dashboard_width, 1200);
    EXPECT_EQ(args.dashboard_height, 720);
    EXPECT_TRUE(args.fullscreen);
    EXPECT_EQ(args.screenshot_dir.string(), std::string("reports/test_live"));
    EXPECT_TRUE(args.capture_output_dir.has_value());
    EXPECT_NEAR(*args.capture_cooldown_sec, 2.5, 1e-12);
    EXPECT_TRUE(args.disable_auto_capture);
    EXPECT_EQ(args.max_frames, 3);
    EXPECT_EQ(args.landmark_backend, double_ok_gesture::LandmarkBackend::OpenCVDebug);
    EXPECT_TRUE(args.landmarks_json.has_value());
    EXPECT_EQ(args.landmarks_json->string(), std::string("configs/debug_landmarks.json"));

    const auto runtime_options = double_ok_gesture::demo_runtime_options(args);
    EXPECT_EQ(runtime_options.camera.source, args.camera.source);
    EXPECT_EQ(runtime_options.config_path.string(), args.config.string());
    EXPECT_EQ(runtime_options.model_path->string(), args.model->string());
    EXPECT_TRUE(runtime_options.threshold.has_value());
    EXPECT_TRUE(runtime_options.require_glasses_pose);
    EXPECT_EQ(runtime_options.capture_output_dir->string(), args.capture_output_dir->string());
    EXPECT_TRUE(runtime_options.capture_cooldown_sec.has_value());
    EXPECT_TRUE(runtime_options.disable_auto_capture);
    EXPECT_EQ(runtime_options.landmark_backend, args.landmark_backend);
    EXPECT_EQ(runtime_options.landmarks_json_path->string(), args.landmarks_json->string());

    const auto frame_options = double_ok_gesture::demo_process_frame_options(parse_demo_args_for_test({
        "double-ok-demo",
        "--capture-gate",
    }));
    EXPECT_TRUE(frame_options.capture_gate);
    EXPECT_FALSE(frame_options.glasses_pose.has_value());
}

void test_demo_args_list_cameras_stops_parsing() {
    const auto args = parse_demo_args_for_test({"double-ok-demo", "--list-cameras", "--unknown"});
    EXPECT_TRUE(args.list_cameras);
}

void test_demo_args_reject_missing_value() {
    bool threw = false;
    try {
        (void)parse_demo_args_for_test({"double-ok-demo", "--camera"});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

void test_null_landmark_provider_returns_empty() {
    double_ok_gesture::NullHandLandmarkProvider provider("rknn");
    const cv::Mat frame(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
    EXPECT_EQ(provider.info().name, std::string("rknn"));
    EXPECT_FALSE(provider.info().available);
    EXPECT_TRUE(provider.detect(frame).empty());
}

void test_json_landmark_provider_reads_real_21_point_hands() {
    double_ok_gesture::JsonHandLandmarkProvider provider(std::filesystem::path("..") / "configs/debug_landmarks.json");
    const cv::Mat frame(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto hands = provider.detect(frame);

    EXPECT_EQ(provider.info().name, std::string("landmarks-json"));
    EXPECT_TRUE(provider.info().available);
    EXPECT_FALSE(provider.info().estimated_landmarks);
    EXPECT_EQ(hands.size(), 2U);
    EXPECT_EQ(hands[0].handedness, std::string("Left"));
    EXPECT_TRUE(hands[0].ok_score.has_value());
    EXPECT_FALSE(hands[0].landmarks_estimated);
    EXPECT_NEAR(hands[0].landmarks[0].x, 0.34, 1e-12);
}

void test_capture_writer_skips_when_not_ready() {
    double_ok_gesture::DataCaptureConfig config;
    config.output_dir = std::filesystem::temp_directory_path() / "double_ok_cpp_capture_writer";
    double_ok_gesture::CaptureWriter writer(config);
    const cv::Mat frame(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto result = make_result({make_hand(0.4, 0.5), make_hand(0.6, 0.5)});
    double_ok_gesture::CaptureGateDecision decision;
    decision.ready = false;

    EXPECT_FALSE(writer.maybe_save(frame, result, decision, "opencv-heuristic").has_value());
    EXPECT_EQ(writer.saved_count(), 0);
}

void test_report_handles_missing_csv() {
    const auto summary = double_ok_gesture::scan_feature_csv(
        std::filesystem::temp_directory_path() / "double_ok_missing_features.csv");
    EXPECT_FALSE(summary.file.exists);
    EXPECT_EQ(summary.row_count, 0U);
    EXPECT_EQ(summary.feature_count, 0U);
}

void test_report_html_contains_key_sections() {
    double_ok_gesture::CsvSummary csv;
    csv.file.path = "data/processed/hagrid_ok_features.csv";
    csv.file.exists = true;
    csv.file.size_label = "1 KB";
    csv.row_count = 2;
    csv.positive_count = 1;
    csv.negative_count = 1;
    csv.feature_count = 96;
    csv.column_count = 102;
    csv.split_counts["train"] = 2;
    csv.gesture_counts["ok"] = 1;
    csv.gesture_counts["palm"] = 1;

    double_ok_gesture::FileSummary model;
    model.path = "models/ok_hand_numpy_logreg.pkl";
    model.exists = true;
    model.size_label = "2 KB";

    const auto html = double_ok_gesture::render_gui_report(
        "configs/default.json",
        double_ok_gesture::RuntimeConfig{},
        csv,
        model);
    EXPECT_TRUE(html.find("Double OK GUI") != std::string::npos);
    EXPECT_TRUE(html.find("21 点关键点示意") != std::string::npos);
    EXPECT_TRUE(html.find("MediaPipe 21 点手部关键点示意") != std::string::npos);
    EXPECT_TRUE(html.find("门控模拟器") != std::string::npos);
}

void test_config_and_pose_loading() {
    const auto path = std::filesystem::temp_directory_path() / "double_ok_cpp_config.json";
    {
        std::ofstream out(path);
        out << R"({"ok_threshold":0.7,"capture_gate":{"require_glasses_pose":true,"center_x_min":0.25,"center_x_max":0.75},"data_capture":{"output_dir":"data/raw/test_gate","cooldown_sec":2.5}})";
    }
    const auto config = double_ok_gesture::load_runtime_config(path);
    EXPECT_NEAR(config.recognizer.ok_threshold, 0.7, 1e-12);
    EXPECT_TRUE(config.capture_gate.require_glasses_pose);
    EXPECT_NEAR(config.capture_gate.center_x_min, 0.25, 1e-12);
    EXPECT_EQ(config.data_capture.output_dir.string(), std::string("data/raw/test_gate"));
    EXPECT_NEAR(config.data_capture.cooldown_sec, 2.5, 1e-12);

    const auto pose_path = std::filesystem::temp_directory_path() / "double_ok_cpp_pose.json";
    {
        std::ofstream out(pose_path);
        out << R"({"pitch":0.0,"roll":1.0,"yaw":-2.0})";
    }
    const auto pose = double_ok_gesture::load_glasses_pose(pose_path);
    EXPECT_TRUE(pose.has_value());
    EXPECT_NEAR(*pose->roll, 1.0, 1e-12);
}

void test_json_parser_reads_hagrid_shape() {
    const auto json = double_ok_gesture::parse_json(
        R"({"image_001":{"label":"ok","hand_landmarks":[[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]]}})");
    EXPECT_TRUE(json.is_object());
    const auto* item = json.get("image_001");
    EXPECT_TRUE(item != nullptr);
    EXPECT_EQ(item->get("label")->as_string(), std::string("ok"));
    EXPECT_EQ(item->get("hand_landmarks")->as_array().front().as_array().size(), 21U);
}

void test_training_and_model_io() {
    const auto csv = std::filesystem::temp_directory_path() / "double_ok_cpp_features.csv";
    {
        std::ofstream out(csv);
        out << "split,target,lm_0_x,lm_0_y\n";
        out << "train,0,0,0\n";
        out << "train,0,0.1,0\n";
        out << "train,1,1,1\n";
        out << "train,1,1.1,1\n";
    }
    const auto dataset = double_ok_gesture::load_feature_csv(csv);
    const auto split = double_ok_gesture::split_data(dataset, 42);
    const auto model = double_ok_gesture::train_logistic_regression(split.x_train, split.y_train, dataset.feature_columns, 20);
    const auto model_path = std::filesystem::temp_directory_path() / "double_ok_cpp_model.txt";
    double_ok_gesture::save_model_artifact(model_path, model);
    const auto loaded = double_ok_gesture::load_model_artifact(model_path);
    EXPECT_EQ(loaded.coef.size(), model.coef.size());
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests = {
        {"feature_vector_has_stable_shape", test_feature_vector_has_stable_shape},
        {"rule_score_prefers_ok_over_open_palm", test_rule_score_prefers_ok_over_open_palm},
        {"capture_gate_ready_and_blocks", test_capture_gate_ready_and_blocks},
        {"capture_gate_requires_double_ok_and_centered", test_capture_gate_requires_double_ok_and_centered},
        {"capture_gate_requires_stable_double_ok", test_capture_gate_requires_stable_double_ok},
        {"negative_capture_gate", test_negative_capture_gate},
        {"recognizer_stability", test_recognizer_stability},
        {"opencv_hand_detector_finds_skin_colored_regions", test_opencv_hand_detector_finds_skin_colored_regions},
        {"estimated_hands_do_not_draw_fake_keypoint_skeleton", test_estimated_hands_do_not_draw_fake_keypoint_skeleton},
        {"dashboard_renders_compact_and_regular_layouts", test_dashboard_renders_compact_and_regular_layouts},
        {"runtime_metrics", test_runtime_metrics},
        {"landmark_backend_parser", test_landmark_backend_parser},
        {"demo_args_parse_and_map_options", test_demo_args_parse_and_map_options},
        {"demo_args_list_cameras_stops_parsing", test_demo_args_list_cameras_stops_parsing},
        {"demo_args_reject_missing_value", test_demo_args_reject_missing_value},
        {"null_landmark_provider_returns_empty", test_null_landmark_provider_returns_empty},
        {"json_landmark_provider_reads_real_21_point_hands", test_json_landmark_provider_reads_real_21_point_hands},
        {"capture_writer_skips_when_not_ready", test_capture_writer_skips_when_not_ready},
        {"report_handles_missing_csv", test_report_handles_missing_csv},
        {"report_html_contains_key_sections", test_report_html_contains_key_sections},
        {"config_and_pose_loading", test_config_and_pose_loading},
        {"json_parser_reads_hagrid_shape", test_json_parser_reads_hagrid_shape},
        {"training_and_model_io", test_training_and_model_io},
    };

    int failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& exc) {
            ++failed;
            std::cerr << "[FAIL] " << name << ": " << exc.what() << '\n';
        }
    }
    return failed == 0 ? 0 : 1;
}
