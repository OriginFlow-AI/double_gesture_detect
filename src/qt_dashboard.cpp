#include "double_ok_gesture/qt_dashboard.hpp"

#include <QApplication>
#include <QDateTime>
#include <QImage>
#include <QKeySequence>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QShortcut>
#include <QSizePolicy>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
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

std::string model_label_for(const QtDashboardOptions& options) {
    if (options.model_path) {
        return options.model_path->filename().string();
    }
    switch (options.landmark_backend) {
        case LandmarkBackend::Rknn:
            return "RKNN 手部关键点";
        case LandmarkBackend::OpenCVDebug:
            return "OpenCV 调试框";
        case LandmarkBackend::None:
            return "未启用关键点";
        case LandmarkBackend::MediaPipe:
            return "MediaPipe 21 点";
        case LandmarkBackend::LandmarksJson:
            return "Landmarks JSON 21 点";
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
        QObject::connect(&timer, &QTimer::timeout, [&]() { update_frame(); });
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
                    model_label_for(options),
                    options.width,
                    options.height));
            }
            return;
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
        const auto snapshot = runtime.metrics.update(frame_result.started);
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
            model_label_for(options),
            options.width,
            options.height);
        present_dashboard(dashboard);

        if (options.max_frames > 0 && frames >= options.max_frames) {
            application.quit();
        }
    }

    int run() {
        timer.start(1);
        if (options.fullscreen) {
            window.showFullScreen();
        } else {
            window.show();
        }
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
