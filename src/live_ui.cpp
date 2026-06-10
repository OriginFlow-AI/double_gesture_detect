#include "double_ok_gesture/live_ui.hpp"

#include <array>
#include <iomanip>
#include <sstream>

namespace double_ok_gesture {
namespace {

const std::array<std::pair<int, int>, 20> kHandConnections = {
    std::pair{0, 1},   std::pair{1, 2},   std::pair{2, 3},   std::pair{3, 4},   std::pair{0, 5},
    std::pair{5, 6},   std::pair{6, 7},   std::pair{7, 8},   std::pair{5, 9},   std::pair{9, 10},
    std::pair{10, 11}, std::pair{11, 12}, std::pair{9, 13},  std::pair{13, 14}, std::pair{14, 15},
    std::pair{15, 16}, std::pair{13, 17}, std::pair{17, 18}, std::pair{18, 19}, std::pair{19, 20},
};

cv::Scalar ok_color(bool ok) {
    return ok ? cv::Scalar(80, 220, 80) : cv::Scalar(60, 180, 255);
}

std::string fixed(double value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

void put_text(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar color, int thickness = 1) {
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

cv::Rect hand_box(const HandPrediction& hand, const cv::Size& size) {
    int x_min = size.width;
    int y_min = size.height;
    int x_max = 0;
    int y_max = 0;
    for (const auto& point : hand.landmarks) {
        const int x = static_cast<int>(std::lround(point.x * size.width));
        const int y = static_cast<int>(std::lround(point.y * size.height));
        x_min = std::min(x_min, x);
        y_min = std::min(y_min, y);
        x_max = std::max(x_max, x);
        y_max = std::max(y_max, y);
    }
    const int pad = 14;
    return cv::Rect(
               cv::Point(std::max(0, x_min - pad), std::max(0, y_min - pad)),
               cv::Point(std::min(size.width - 1, x_max + pad), std::min(size.height - 1, y_max + pad))) &
           cv::Rect(0, 0, size.width, size.height);
}

void draw_status_row(cv::Mat& panel, int y, const std::string& label, bool passed) {
    const cv::Scalar color = passed ? cv::Scalar(80, 220, 80) : cv::Scalar(60, 180, 255);
    cv::circle(panel, {24, y - 5}, 8, color, -1, cv::LINE_AA);
    put_text(panel, label, {44, y}, 0.45, cv::Scalar(235, 235, 235), 1);
    put_text(panel, passed ? "PASS" : "WAIT", {250, y}, 0.42, color, 1);
}

cv::Mat fit_image(const cv::Mat& source, const cv::Size& target) {
    cv::Mat resized;
    const double scale = std::min(
        static_cast<double>(target.width) / static_cast<double>(source.cols),
        static_cast<double>(target.height) / static_cast<double>(source.rows));
    const int width = std::max(1, static_cast<int>(std::lround(source.cols * scale)));
    const int height = std::max(1, static_cast<int>(std::lround(source.rows * scale)));
    cv::resize(source, resized, {width, height});
    cv::Mat canvas(target, source.type(), cv::Scalar(18, 22, 26));
    const int x = (target.width - width) / 2;
    const int y = (target.height - height) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, width, height)));
    return canvas;
}

}  // namespace

void draw_hand_tracking(cv::Mat& frame_bgr, const DoubleOKResult& result) {
    const cv::Size size = frame_bgr.size();
    for (const auto& hand : result.hands) {
        const cv::Scalar color = ok_color(hand.is_ok);
        if (hand.landmarks_estimated) {
            const cv::Rect box = hand_box(hand, size);
            cv::rectangle(frame_bgr, box, color, 2, cv::LINE_AA);
            put_text(
                frame_bgr,
                hand.handedness + " candidate " + fixed(hand.ok_score * 100.0, 1) + "%",
                {box.x, std::max(20, box.y - 8)},
                0.52,
                color,
                2);
            put_text(
                frame_bgr,
                "OpenCV heuristic, not MediaPipe landmarks",
                {box.x, std::min(size.height - 10, box.y + box.height + 22)},
                0.42,
                cv::Scalar(80, 210, 255),
                1);
            continue;
        }
        std::array<cv::Point, 21> points{};
        for (std::size_t i = 0; i < hand.landmarks.size(); ++i) {
            points[i] = {
                static_cast<int>(std::lround(hand.landmarks[i].x * size.width)),
                static_cast<int>(std::lround(hand.landmarks[i].y * size.height)),
            };
        }
        for (const auto& [start, end] : kHandConnections) {
            cv::line(frame_bgr, points[start], points[end], color, 2, cv::LINE_AA);
        }
        for (const auto& point : points) {
            cv::circle(frame_bgr, point, 4, cv::Scalar(245, 245, 245), -1, cv::LINE_AA);
            cv::circle(frame_bgr, point, 5, color, 1, cv::LINE_AA);
        }
        const cv::Rect box = hand_box(hand, size);
        cv::rectangle(frame_bgr, box, color, 2, cv::LINE_AA);
        put_text(
            frame_bgr,
            hand.handedness + " " + fixed(hand.ok_score * 100.0, 1) + "%",
            {box.x, std::max(20, box.y - 8)},
            0.52,
            color,
            2);
    }
}

void draw_capture_guides(cv::Mat& frame_bgr, const CaptureGateDecision& decision, const CaptureGateConfig& config) {
    const cv::Scalar color = decision.ready ? cv::Scalar(80, 220, 80) : cv::Scalar(60, 180, 255);
    const int x1 = static_cast<int>(config.center_x_min * frame_bgr.cols);
    const int x2 = static_cast<int>(config.center_x_max * frame_bgr.cols);
    const int y1 = static_cast<int>(config.center_y_min * frame_bgr.rows);
    const int y2 = static_cast<int>(config.center_y_max * frame_bgr.rows);
    cv::rectangle(frame_bgr, {x1, y1}, {x2, y2}, color, 2, cv::LINE_AA);
    put_text(frame_bgr, gate_reason_value(decision.reason), {x1 + 8, std::max(24, y1 - 10)}, 0.58, color, 2);
}

cv::Mat render_dashboard(
    const cv::Mat& camera_frame,
    const DoubleOKResult& result,
    const std::optional<CaptureGateDecision>& decision,
    const RuntimeSnapshot& snapshot,
    const std::string& camera_label,
    double target_fps,
    const std::string& model_label,
    int width,
    int height) {
    width = std::max(width, 960);
    height = std::max(height, 600);
    const int sidebar_w = 350;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(24, 24, 24));
    const cv::Rect image_rect(0, 0, width - sidebar_w, height);
    cv::Mat fitted = fit_image(camera_frame, image_rect.size());
    fitted.copyTo(canvas(image_rect));

    cv::Mat panel = canvas(cv::Rect(width - sidebar_w, 0, sidebar_w, height));
    panel.setTo(cv::Scalar(36, 32, 28));
    const bool ready = decision && decision->ready;
    const cv::Scalar state_color = ready ? cv::Scalar(80, 220, 80) : cv::Scalar(60, 180, 255);
    put_text(panel, ready ? "READY" : "WAITING", {20, 42}, 0.85, state_color, 2);
    put_text(panel, "Double OK C++", {20, 74}, 0.48, cv::Scalar(190, 190, 190), 1);
    put_text(panel, "FPS " + fixed(snapshot.fps, 1), {20, 118}, 0.58, cv::Scalar(235, 235, 235), 1);
    put_text(panel, "PROC " + fixed(snapshot.processing_ms, 1) + " ms", {150, 118}, 0.58, cv::Scalar(235, 235, 235), 1);
    put_text(panel, "CAM " + camera_label, {20, 152}, 0.42, cv::Scalar(210, 210, 210), 1);
    put_text(panel, "MODEL " + model_label, {20, 180}, 0.42, cv::Scalar(210, 210, 210), 1);

    int y = 228;
    if (decision) {
        draw_status_row(panel, y, "POSE", decision->glasses_pose_ok);
        y += 34;
        draw_status_row(panel, y, "VISIBLE", decision->hands_visible);
        y += 34;
        draw_status_row(panel, y, "CENTERED", decision->hands_centered);
        y += 34;
        draw_status_row(panel, y, "SEPARATED", decision->hands_separated);
        y += 34;
        draw_status_row(panel, y, "GESTURE", decision->gesture_ok);
        y += 52;
        put_text(panel, decision->prompt, {20, y}, 0.46, state_color, 1);
    } else {
        put_text(panel, "Capture gate disabled", {20, y}, 0.48, cv::Scalar(210, 210, 210), 1);
        y += 52;
    }

    y += 44;
    put_text(panel, "Hands", {20, y}, 0.58, cv::Scalar(235, 235, 235), 1);
    y += 36;
    for (std::size_t i = 0; i < 2; ++i) {
        if (i < result.hands.size()) {
            const auto& hand = result.hands[i];
            const cv::Scalar color = ok_color(hand.is_ok);
            put_text(panel, hand.handedness, {20, y}, 0.48, cv::Scalar(235, 235, 235), 1);
            put_text(panel, fixed(hand.ok_score * 100.0, 1) + "%", {165, y}, 0.48, color, 1);
            put_text(panel, hand.is_ok ? "OK" : "NOT OK", {250, y}, 0.48, color, 1);
        } else {
            put_text(panel, "HAND " + std::to_string(i + 1) + " NOT DETECTED", {20, y}, 0.44, cv::Scalar(155, 155, 155), 1);
        }
        y += 34;
    }

    const cv::Scalar fps_color = snapshot.frame_count < 10 || target_fps <= 0.0 || snapshot.fps >= target_fps
                                     ? cv::Scalar(80, 220, 80)
                                     : cv::Scalar(60, 180, 255);
    put_text(panel, "Q/ESC exit, S screenshot", {20, height - 26}, 0.42, fps_color, 1);
    return canvas;
}

}  // namespace double_ok_gesture
