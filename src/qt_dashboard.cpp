#include "double_ok_gesture/qt_dashboard.hpp"

#include <QApplication>
#include <QDateTime>
#include <QImage>
#include <QKeySequence>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QScreen>
#include <QShortcut>
#include <QSizePolicy>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "double_ok_gesture/capture_writer.hpp"
#include "double_ok_gesture/live_ui.hpp"

namespace double_ok_gesture {
namespace {

QImage mat_to_image(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

QString camera_label(const CameraStream& camera) {
    const auto& info = camera.info();
    return QStringLiteral("%1 | %2x%3 | %4 帧/秒 | %5")
        .arg(QString::fromStdString(info.source))
        .arg(info.width)
        .arg(info.height)
        .arg(info.fps, 0, 'f', 1)
        .arg(QString::fromStdString(info.fourcc));
}

std::string model_label_for(
    const QtDashboardOptions& options,
    const RuntimeConfig& config) {
    if (options.landmark_backend == LandmarkBackend::Onnx) {
        const auto palm = options.palm_model_path.value_or(
            config.onnx_hand.palm_model_path);
        const auto hand = options.hand_model_path.value_or(
            config.onnx_hand.hand_model_path);
        return "MediaPipe ONNX FP32 · " + palm.filename().string() +
               " + " + hand.filename().string();
    }
    if (options.landmark_backend == LandmarkBackend::Rknn) {
        const auto palm = options.palm_model_path.value_or(
            config.rknn_hand.palm_model_path);
        const auto hand = options.hand_model_path.value_or(
            config.rknn_hand.hand_model_path);
        return "MediaPipe RKNN FP16 · " + palm.filename().string() +
               " + " + hand.filename().string();
    }
    switch (options.landmark_backend) {
        case LandmarkBackend::Rknn:
            return "MediaPipe / RKNN NPU";
        case LandmarkBackend::Onnx:
            return "MediaPipe / ONNX";
        case LandmarkBackend::None:
            return "未启用关键点";
        case LandmarkBackend::LandmarksJson:
            return "JSON 测试后端";
    }
    return "几何规则";
}

}  // namespace

struct QtDashboard::Impl {
    Impl(QApplication& application, RuntimeBundle& runtime, QtDashboardOptions options)
        : application(application), runtime(runtime), options(std::move(options)),
          capture_writer(runtime.config.data_capture) {
        build_ui();
        connect_actions();
    }

    QApplication& application;
    RuntimeBundle& runtime;
    QtDashboardOptions options;
    QMainWindow window;
    QTimer timer;
    CaptureWriter capture_writer;
    cv::Mat last_rendered_frame;
    int frames = 0;
    bool stopping = false;

    QLabel* video_label = nullptr;

    ProcessFrameOptions process_frame_options() const {
        ProcessFrameOptions frame_options;
        frame_options.capture_gate = options.capture_gate;
        if (options.glasses_pose) {
            frame_options.glasses_pose = load_glasses_pose(*options.glasses_pose);
        }
        return frame_options;
    }

    void build_ui() {
        window.setWindowTitle(QStringLiteral("双手 OK 采集门控"));
        window.resize(options.width, options.height);

        auto* central = new QWidget(&window);
        auto* root = new QVBoxLayout(central);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);
        video_label = new QLabel(QStringLiteral("等待视频帧"));
        video_label->setObjectName("DashboardCanvas");
        video_label->setAlignment(Qt::AlignCenter);
        video_label->setMinimumSize(options.width, options.height);
        video_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        root->addWidget(video_label, 1);
        window.setCentralWidget(central);
        window.setStyleSheet(R"(
          QMainWindow, QWidget, QLabel {
            background: #1e1e1e;
            color: #f2f2f2;
            font-family: "Noto Sans CJK SC", "Microsoft YaHei", "Sans Serif";
          }
          #DashboardCanvas {
            background: #1e1e1e;
            color: #a9b0bb;
            font-weight: 600;
          }
        )");

        auto* quit_shortcut = new QShortcut(QKeySequence(Qt::Key_Q), &window);
        QObject::connect(quit_shortcut, &QShortcut::activated, &application, &QApplication::quit);
        auto* escape_shortcut = new QShortcut(QKeySequence(Qt::Key_Escape), &window);
        QObject::connect(escape_shortcut, &QShortcut::activated, &application, &QApplication::quit);
        auto* screenshot_shortcut = new QShortcut(QKeySequence(Qt::Key_S), &window);
        QObject::connect(screenshot_shortcut, &QShortcut::activated, [&]() { save_screenshot(); });
    }

    void connect_actions() {
        timer.setSingleShot(true);
        QObject::connect(&timer, &QTimer::timeout, [&]() {
            update_frame();
            if (!stopping) {
                timer.start(frame_interval_ms());
            }
        });
    }

    int frame_interval_ms() const {
        if (options.target_fps <= 0.0) {
            return 1;
        }
        return std::max(
            1,
            static_cast<int>(std::lround(1000.0 / options.target_fps)));
    }

    void present_dashboard(const cv::Mat& dashboard) {
        last_rendered_frame = dashboard.clone();
        const QImage image = mat_to_image(dashboard);
        const QSize target_size = video_label->size().isEmpty() ? QSize(options.width, options.height) : video_label->size();
        video_label->setPixmap(QPixmap::fromImage(image).scaled(
            target_size,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
    }

    void save_screenshot() {
        if (last_rendered_frame.empty()) {
            return;
        }
        std::filesystem::create_directories(options.screenshot_dir);
        const auto path =
            options.screenshot_dir /
            ("double_ok_qt_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss").toStdString() + ".png");
        if (cv::imwrite(path.string(), last_rendered_frame)) {
            window.setWindowTitle(QStringLiteral("双手 OK 采集门控 · 截图已保存 %1").arg(QString::fromStdString(path.string())));
        }
    }

    void update_frame() {
        auto frame = runtime.camera.read();
        if (!frame) {
            if (last_rendered_frame.empty()) {
                present_dashboard(render_dashboard(
                    cv::Mat{},
                    DoubleOKResult{},
                    std::nullopt,
                    RuntimeSnapshot{0.0, 0.0, 0},
                    camera_label(runtime.camera).toStdString(),
                    options.target_fps,
                    model_label_for(options, runtime.config),
                    options.width,
                    options.height));
            }
            return;
        }

        if (options.right_half && frame->cols == 3840 && frame->rows == 1080) {
            *frame = frame->operator()(cv::Rect(options.crop_x, options.crop_y, options.crop_width, options.crop_height)).clone();
        }

        const auto frame_result = process_runtime_frame(runtime, *frame, process_frame_options());
        const auto& result = frame_result.result;
        const auto& decision = frame_result.decision;
        if (decision) {
            if (auto saved = capture_writer.maybe_save(
                    *frame,
                    result,
                    *decision,
                    landmark_backend_value(options.landmark_backend))) {
                window.setWindowTitle(QStringLiteral("双手 OK 采集门控 · 已采集 %1").arg(QString::fromStdString(saved->filename().string())));
            }
        }
        auto snapshot = runtime.metrics.update(frame_result.started);
        snapshot.inference_ms = frame_result.inference_ms;
        ++frames;

        cv::Mat display = frame->clone();
        draw_hand_tracking(display, result);
        if (decision) {
            draw_capture_guides(display, *decision, runtime.config.capture_gate);
        }
        cv::Mat dashboard = render_dashboard(
            display,
            result,
            decision,
            snapshot,
            camera_label(runtime.camera).toStdString(),
            options.target_fps,
            model_label_for(options, runtime.config),
            options.width,
            options.height);
        present_dashboard(dashboard);

        if (options.max_frames > 0 && frames >= options.max_frames) {
            stopping = true;
            application.quit();
        }
    }

    int run() {
        timer.start(0);
        if (options.fullscreen) {
            window.showFullScreen();
        } else {
            if (const QScreen* screen = application.primaryScreen()) {
                const QRect available = screen->availableGeometry();
                window.move(
                    available.x() + (available.width() - window.width()) / 2,
                    available.y() + (available.height() - window.height()) / 2);
            }
            window.show();
        }
        window.raise();
        window.activateWindow();
        return application.exec();
    }
};

QtDashboard::QtDashboard(QApplication& application, RuntimeBundle& runtime, QtDashboardOptions options)
    : impl_(std::make_unique<Impl>(application, runtime, std::move(options))) {}

QtDashboard::~QtDashboard() = default;

int QtDashboard::run() {
    return impl_->run();
}

}  // namespace double_ok_gesture
