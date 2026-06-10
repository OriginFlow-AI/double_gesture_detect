#include <QApplication>
#include <QDateTime>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QSizePolicy>
#include <QStatusBar>
#include <QStyle>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "double_ok_gesture/camera.hpp"
#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/hand_detector.hpp"
#include "double_ok_gesture/live_ui.hpp"
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
    double target_fps = 25.0;
    int dashboard_width = 1440;
    int dashboard_height = 810;
    bool fullscreen = false;
    std::filesystem::path screenshot_dir = "reports/live";
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
        } else if (key == "--voice-prompts") {
            // Kept for CLI compatibility. TTS is intentionally not spawned in the C++ build.
        } else if (key == "--prompt-interval") {
            (void)next();
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

QFrame* makePanel(const QString& object_name = QStringLiteral("Panel")) {
    auto* panel = new QFrame();
    panel->setObjectName(object_name);
    return panel;
}

QLabel* makeSectionTitle(const QString& text) {
    auto* label = new QLabel(text);
    label->setObjectName("SectionTitle");
    return label;
}

QLabel* makeMetaLabel(const QString& text) {
    auto* label = new QLabel(text);
    label->setObjectName("MetaLabel");
    label->setWordWrap(true);
    label->setMinimumHeight(34);
    return label;
}

QTextEdit* makeReadOnlyText(const QString& object_name, int min_height) {
    auto* edit = new QTextEdit();
    edit->setObjectName(object_name);
    edit->setReadOnly(true);
    edit->setMinimumHeight(min_height);
    edit->setLineWrapMode(QTextEdit::WidgetWidth);
    return edit;
}

void polish(QWidget* widget) {
    widget->style()->unpolish(widget);
    widget->style()->polish(widget);
}

void setState(QLabel* label, const QString& text, const QString& state) {
    label->setText(text);
    label->setProperty("state", state);
    polish(label);
}

QString yesNo(bool value) {
    return value ? QStringLiteral("通过") : QStringLiteral("等待");
}

QString fmt(double value, int precision = 1) {
    return QString::number(value, 'f', precision);
}

QImage matToImage(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

QString cameraLabel(const double_ok_gesture::CameraStream& camera) {
    const auto& info = camera.info();
    return QStringLiteral("%1 · %2x%3 · %4 FPS · %5")
        .arg(QString::fromStdString(info.source))
        .arg(info.width)
        .arg(info.height)
        .arg(info.fps, 0, 'f', 1)
        .arg(QString::fromStdString(info.fourcc));
}

void appendEvent(QTextEdit* event_text, const QString& level, const QString& message) {
    event_text->append(
        QStringLiteral("[%1] %2  %3")
            .arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")))
            .arg(level)
            .arg(message));
}

struct RuntimeBundle {
    double_ok_gesture::RuntimeConfig config;
    double_ok_gesture::OKHandClassifier classifier;
    double_ok_gesture::DoubleOKRecognizer recognizer;
    double_ok_gesture::OpenCVHandDetector detector;
    double_ok_gesture::CameraStream camera;
    double_ok_gesture::RuntimeMetrics metrics;
};

RuntimeBundle makeRuntime(const Args& args) {
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
    double_ok_gesture::OpenCVHandDetector detector({
        runtime_config.recognizer.max_num_hands,
        0.006,
        3.0,
    });
    return {
        runtime_config,
        classifier,
        recognizer,
        detector,
        double_ok_gesture::open_camera(args.camera),
        double_ok_gesture::RuntimeMetrics(),
    };
}

int runHeadless(const Args& args) {
    RuntimeBundle runtime = makeRuntime(args);
    double last_status = 0.0;
    int frames = 0;
    while (true) {
        auto frame = runtime.camera.read();
        if (!frame) {
            continue;
        }
        const double started = double_ok_gesture::monotonic_seconds();
        const auto detected_hands = runtime.detector.detect(*frame);
        const auto result = runtime.recognizer.process_hands(detected_hands);
        std::optional<double_ok_gesture::CaptureGateDecision> decision;
        if (args.capture_gate) {
            const auto pose = args.glasses_pose ? double_ok_gesture::load_glasses_pose(*args.glasses_pose) : std::nullopt;
            decision = double_ok_gesture::evaluate_capture_gate(result, runtime.config.capture_gate, pose);
        }
        const auto snapshot = runtime.metrics.update(started);
        ++frames;

        const double now = double_ok_gesture::monotonic_seconds();
        if (args.status_interval == 0.0 || now - last_status >= args.status_interval) {
            std::cout << "hands=" << result.hands.size() << " ok_count=" << result.ok_count
                      << " double_ok=" << result.double_ok << " stable=" << result.stable_double_ok;
            if (decision) {
                std::cout << " gate_ready=" << decision->ready << " reason="
                          << double_ok_gesture::gate_reason_value(decision->reason);
            }
            std::cout << " fps=" << snapshot.fps << " processing_ms=" << snapshot.processing_ms << '\n';
            last_status = now;
        }
        if (args.max_frames > 0 && frames >= args.max_frames) {
            break;
        }
    }
    runtime.camera.close();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.headless) {
            return runHeadless(args);
        }

        QApplication application(argc, argv);
        RuntimeBundle runtime = makeRuntime(args);

        QMainWindow window;
        window.setWindowTitle(QStringLiteral("双手 OK 采集门控"));
        window.resize(args.dashboard_width, args.dashboard_height);

        auto* central = new QWidget(&window);
        auto* root = new QVBoxLayout(central);
        root->setContentsMargins(16, 14, 16, 16);
        root->setSpacing(12);

        auto* header_panel = makePanel(QStringLiteral("HeaderPanel"));
        auto* header = new QHBoxLayout(header_panel);
        header->setContentsMargins(16, 14, 16, 14);
        header->setSpacing(14);

        auto* title_block = new QVBoxLayout();
        title_block->setSpacing(4);
        auto* title = new QLabel(QStringLiteral("双手 OK 采集门控"));
        title->setObjectName("AppTitle");
        auto* subtitle = new QLabel(QStringLiteral("C++ 实时检测、双手 OK 稳定判断与采集前门控"));
        subtitle->setObjectName("HintLabel");
        title_block->addWidget(title);
        title_block->addWidget(subtitle);

        auto* session_label = new QLabel(QStringLiteral("实时监控"));
        session_label->setObjectName("SessionLabel");
        session_label->setProperty("state", QStringLiteral("running"));
        auto* camera_label = new QLabel(cameraLabel(runtime.camera));
        camera_label->setObjectName("SensorLabel");
        camera_label->setProperty("state", QStringLiteral("ok"));
        header->addLayout(title_block, 1);
        header->addStretch(1);
        header->addWidget(session_label);
        header->addWidget(camera_label);
        root->addWidget(header_panel);

        auto* body = new QHBoxLayout();
        body->setSpacing(12);

        auto* video_panel = makePanel();
        auto* video_layout = new QVBoxLayout(video_panel);
        video_layout->setContentsMargins(14, 12, 14, 14);
        video_layout->setSpacing(10);
        video_layout->addWidget(makeSectionTitle(QStringLiteral("实时画面")));
        auto* video_label = new QLabel(QStringLiteral("等待视频帧"));
        video_label->setObjectName("FigureLabel");
        video_label->setAlignment(Qt::AlignCenter);
        video_label->setMinimumSize(760, 520);
        video_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        video_layout->addWidget(video_label, 1);
        body->addWidget(video_panel, 5);

        auto* status_panel = makePanel();
        auto* status_layout = new QVBoxLayout(status_panel);
        status_layout->setContentsMargins(14, 12, 14, 14);
        status_layout->setSpacing(10);
        status_layout->addWidget(makeSectionTitle(QStringLiteral("采集状态")));
        auto* workflow_label = makeMetaLabel(QStringLiteral("等待检测双手"));
        workflow_label->setProperty("state", QStringLiteral("idle"));
        auto* detector_label = makeMetaLabel(QStringLiteral("检测后端：OpenCV 候选检测（临时）"));
        detector_label->setProperty("state", QStringLiteral("warning"));
        auto* model_label = makeMetaLabel(
            args.model ? QStringLiteral("模型：%1").arg(QString::fromStdString(args.model->filename().string()))
                       : QStringLiteral("模型：几何规则"));
        model_label->setProperty("state", QStringLiteral("running"));
        status_layout->addWidget(workflow_label);
        status_layout->addWidget(detector_label);
        status_layout->addWidget(model_label);

        auto* readiness_progress = new QProgressBar();
        readiness_progress->setRange(0, 100);
        readiness_progress->setValue(0);
        readiness_progress->setFormat(QStringLiteral("采集条件 0%"));
        readiness_progress->setMinimumHeight(28);
        status_layout->addWidget(readiness_progress);

        auto* gate_text = makeReadOnlyText(QStringLiteral("StatsText"), 170);
        gate_text->setPlainText(QStringLiteral("等待门控结果..."));
        status_layout->addWidget(gate_text);

        status_layout->addWidget(makeSectionTitle(QStringLiteral("手势状态")));
        auto* hands_text = makeReadOnlyText(QStringLiteral("PreviewText"), 135);
        hands_text->setPlainText(QStringLiteral("等待检测..."));
        status_layout->addWidget(hands_text);

        status_layout->addWidget(makeSectionTitle(QStringLiteral("事件日志")));
        auto* event_text = makeReadOnlyText(QStringLiteral("EventText"), 150);
        appendEvent(event_text, QStringLiteral("信息"), QStringLiteral("应用已启动。"));
        appendEvent(
            event_text,
            QStringLiteral("警告"),
            QStringLiteral("当前后端是 OpenCV 启发式检测，生产精度需要接入 MediaPipe C++ landmark。"));
        status_layout->addWidget(event_text, 1);

        auto* controls = new QHBoxLayout();
        auto* screenshot_button = new QPushButton(QStringLiteral("保存截图"));
        screenshot_button->setObjectName("GhostButton");
        auto* quit_button = new QPushButton(QStringLiteral("退出"));
        quit_button->setObjectName("PrimaryButton");
        controls->addWidget(screenshot_button);
        controls->addWidget(quit_button);
        status_layout->addLayout(controls);
        body->addWidget(status_panel, 2);

        root->addLayout(body, 1);
        window.setCentralWidget(central);
        window.statusBar()->showMessage(QStringLiteral("就绪"));

        window.setStyleSheet(R"(
          QMainWindow, QWidget { background: #0a0d11; color: #e8eef5; font-family: "Noto Sans CJK SC", "Microsoft YaHei", "Sans Serif"; font-size: 13px; }
          #HeaderPanel, #Panel { background: #12161d; border: 1px solid #293241; border-radius: 6px; }
          #AppTitle { font-size: 22px; font-weight: 700; color: #f8fafc; }
          #SectionTitle { font-size: 15px; font-weight: 700; color: #f8fafc; }
          #HintLabel { color: #9ba8b8; }
          #SensorLabel, #SessionLabel, #MetaLabel { background: #181d25; border: 1px solid #333d4c; border-radius: 4px; padding: 7px 10px; color: #bac4d0; }
          #MetaLabel[state="idle"] { color: #bac4d0; }
          #MetaLabel[state="running"], #SessionLabel[state="running"] { color: #73d7ff; border-color: #286f8c; }
          #MetaLabel[state="ok"], #SensorLabel[state="ok"] { color: #91eba6; border-color: #2d8249; }
          #MetaLabel[state="warning"] { color: #ffd27a; border-color: #8d6b2b; }
          #MetaLabel[state="bad"], #SensorLabel[state="bad"] { color: #ff967f; border-color: #894d38; }
          QPushButton { background: #1a2029; border: 1px solid #3a4657; border-radius: 4px; padding: 8px 12px; color: #f8fafc; font-weight: 600; }
          QPushButton#PrimaryButton { background: #1b5a60; border-color: #27848d; }
          QPushButton#GhostButton { background: #20222a; border-color: #474b57; }
          QPushButton:hover:!disabled { background: #237179; border-color: #36a6b2; }
          QProgressBar { background: #0d1117; border: 1px solid #2f3948; border-radius: 4px; height: 28px; color: #f8fafc; text-align: center; font-weight: 700; }
          QProgressBar::chunk { background: #d39b40; border-radius: 3px; }
          QTextEdit { background: #0d1117; border: 1px solid #293241; border-radius: 4px; color: #e8eef5; font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 12px; }
          #FigureLabel { background: #0d1117; border: 1px solid #293241; border-radius: 4px; color: #7f8b9a; font-weight: 600; }
          QStatusBar { background: #0a0d11; color: #9ba8b8; }
        )");

        cv::Mat last_rendered_frame;
        int frames = 0;
        QObject::connect(quit_button, &QPushButton::clicked, &application, &QApplication::quit);
        QObject::connect(screenshot_button, &QPushButton::clicked, [&]() {
            if (last_rendered_frame.empty()) {
                return;
            }
            std::filesystem::create_directories(args.screenshot_dir);
            const auto path =
                args.screenshot_dir /
                ("double_ok_qt_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss").toStdString() + ".png");
            if (cv::imwrite(path.string(), last_rendered_frame)) {
                appendEvent(event_text, QStringLiteral("信息"), QStringLiteral("截图已保存：%1").arg(QString::fromStdString(path.string())));
            }
        });

        QTimer timer;
        QObject::connect(&timer, &QTimer::timeout, [&]() {
            auto frame = runtime.camera.read();
            if (!frame) {
                return;
            }
            const double started = double_ok_gesture::monotonic_seconds();
            const auto detected_hands = runtime.detector.detect(*frame);
            const auto result = runtime.recognizer.process_hands(detected_hands);
            std::optional<double_ok_gesture::CaptureGateDecision> decision;
            if (args.capture_gate) {
                const auto pose = args.glasses_pose ? double_ok_gesture::load_glasses_pose(*args.glasses_pose) : std::nullopt;
                decision = double_ok_gesture::evaluate_capture_gate(result, runtime.config.capture_gate, pose);
            }
            const auto snapshot = runtime.metrics.update(started);
            ++frames;

            cv::Mat display = frame->clone();
            double_ok_gesture::draw_hand_tracking(display, result);
            if (decision) {
                double_ok_gesture::draw_capture_guides(display, *decision, runtime.config.capture_gate);
            }
            last_rendered_frame = display.clone();
            const QImage image = matToImage(display);
            video_label->setPixmap(QPixmap::fromImage(image).scaled(
                video_label->size(),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation));

            const int passed = decision
                                   ? static_cast<int>(decision->glasses_pose_ok) + static_cast<int>(decision->hands_visible) +
                                         static_cast<int>(decision->hands_centered) + static_cast<int>(decision->hands_separated) +
                                         static_cast<int>(decision->gesture_ok)
                                   : 0;
            const int percent = decision ? passed * 20 : 0;
            readiness_progress->setValue(percent);
            readiness_progress->setFormat(QStringLiteral("采集条件 %1%").arg(percent));
            if (decision) {
                const QString state = decision->ready ? QStringLiteral("ok") : QStringLiteral("warning");
                setState(
                    workflow_label,
                    QStringLiteral("%1 · %2")
                        .arg(QString::fromUtf8(double_ok_gesture::gate_reason_value(decision->reason)))
                        .arg(QString::fromStdString(decision->prompt)),
                    state);
                gate_text->setPlainText(
                    QStringLiteral("姿态: %1\n完整入框: %2\n中心区域: %3\n双手分离: %4\n手势: %5\nFPS: %6\n处理: %7 ms")
                        .arg(yesNo(decision->glasses_pose_ok))
                        .arg(yesNo(decision->hands_visible))
                        .arg(yesNo(decision->hands_centered))
                        .arg(yesNo(decision->hands_separated))
                        .arg(yesNo(decision->gesture_ok))
                        .arg(fmt(snapshot.fps))
                        .arg(fmt(snapshot.processing_ms)));
            }

            QString hands_summary;
            for (std::size_t i = 0; i < result.hands.size(); ++i) {
                const auto& hand = result.hands[i];
                hands_summary += QStringLiteral("%1  %2  %3%  %4\n")
                                     .arg(QString::fromStdString(hand.handedness))
                                     .arg(hand.is_ok ? QStringLiteral("OK") : QStringLiteral("NOT OK"))
                                     .arg(hand.ok_score * 100.0, 0, 'f', 1)
                                     .arg(hand.landmarks_estimated ? QStringLiteral("候选框") : QStringLiteral("21点"));
            }
            if (hands_summary.isEmpty()) {
                hands_summary = QStringLiteral("未检测到手部候选");
            }
            hands_text->setPlainText(hands_summary);
            window.statusBar()->showMessage(
                QStringLiteral("hands=%1 ok=%2 fps=%3 proc=%4ms")
                    .arg(result.hands.size())
                    .arg(result.ok_count)
                    .arg(fmt(snapshot.fps))
                    .arg(fmt(snapshot.processing_ms)));

            if (args.max_frames > 0 && frames >= args.max_frames) {
                application.quit();
            }
        });

        timer.start(1);
        if (args.fullscreen) {
            window.showFullScreen();
        } else {
            window.show();
        }
        const int exit_code = application.exec();
        runtime.camera.close();
        return exit_code;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
