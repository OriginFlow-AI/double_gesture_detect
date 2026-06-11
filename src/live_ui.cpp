#include "double_ok_gesture/live_ui.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace double_ok_gesture {
namespace {

const std::array<std::pair<int, int>, 21> kHandConnections = {
    std::pair{0, 1},   std::pair{1, 2},   std::pair{2, 3},   std::pair{3, 4},   std::pair{0, 5},
    std::pair{5, 6},   std::pair{6, 7},   std::pair{7, 8},   std::pair{5, 9},   std::pair{9, 10},
    std::pair{10, 11}, std::pair{11, 12}, std::pair{9, 13},  std::pair{13, 14}, std::pair{14, 15},
    std::pair{15, 16}, std::pair{13, 17}, std::pair{17, 18}, std::pair{18, 19}, std::pair{19, 20},
    std::pair{0, 17},
};

const cv::Scalar kBackground(30, 30, 30);
const cv::Scalar kTopbar(31, 31, 31);
const cv::Scalar kPanel(38, 38, 38);
const cv::Scalar kPanelAlt(48, 48, 48);
const cv::Scalar kBorder(68, 68, 68);
const cv::Scalar kText(238, 238, 238);
const cv::Scalar kMuted(170, 170, 170);
const cv::Scalar kAccent(245, 139, 76);
const cv::Scalar kSuccess(107, 194, 74);
const cv::Scalar kWarning(68, 160, 240);
const cv::Scalar kDanger(80, 83, 239);

cv::Scalar ok_color(bool ok) {
    return ok ? kSuccess : kAccent;
}

std::string fixed(double value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

void put_text(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar color, int thickness = 1) {
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

void rounded_rect(cv::Mat& image, const cv::Rect& rect, cv::Scalar color, int radius, std::optional<cv::Scalar> border = std::nullopt) {
    if (rect.width <= 0 || rect.height <= 0) {
        return;
    }
    radius = std::max(1, std::min({radius, rect.width / 2, rect.height / 2}));
    cv::rectangle(image, {rect.x + radius, rect.y}, {rect.x + rect.width - radius, rect.y + rect.height}, color, -1);
    cv::rectangle(image, {rect.x, rect.y + radius}, {rect.x + rect.width, rect.y + rect.height - radius}, color, -1);
    for (const cv::Point center : {
             cv::Point(rect.x + radius, rect.y + radius),
             cv::Point(rect.x + rect.width - radius, rect.y + radius),
             cv::Point(rect.x + radius, rect.y + rect.height - radius),
             cv::Point(rect.x + rect.width - radius, rect.y + rect.height - radius),
         }) {
        cv::circle(image, center, radius, color, -1, cv::LINE_AA);
    }
    if (border) {
        cv::line(image, {rect.x + radius, rect.y}, {rect.x + rect.width - radius, rect.y}, *border, 1, cv::LINE_AA);
        cv::line(
            image,
            {rect.x + radius, rect.y + rect.height},
            {rect.x + rect.width - radius, rect.y + rect.height},
            *border,
            1,
            cv::LINE_AA);
        cv::line(image, {rect.x, rect.y + radius}, {rect.x, rect.y + rect.height - radius}, *border, 1, cv::LINE_AA);
        cv::line(
            image,
            {rect.x + rect.width, rect.y + radius},
            {rect.x + rect.width, rect.y + rect.height - radius},
            *border,
            1,
            cv::LINE_AA);
    }
}

void right_text(cv::Mat& image, const std::string& text, cv::Point anchor, double scale, cv::Scalar color, int thickness = 1) {
    const int width = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, nullptr).width;
    put_text(image, text, {anchor.x - width, anchor.y}, scale, color, thickness);
}

void centered_text(cv::Mat& image, const std::string& text, const cv::Rect& rect, double scale, cv::Scalar color, int thickness = 1) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    const cv::Point origin(rect.x + (rect.width - size.width) / 2, rect.y + (rect.height + size.height - baseline) / 2);
    put_text(image, text, origin, scale, color, thickness);
}

void progress_bar(cv::Mat& image, const cv::Rect& rect, double progress, cv::Scalar color, cv::Scalar track) {
    progress = std::clamp(progress, 0.0, 1.0);
    rounded_rect(image, rect, track, std::max(2, rect.height / 2));
    const int fill_width = static_cast<int>(std::lround(static_cast<double>(rect.width) * progress));
    if (fill_width > 0) {
        rounded_rect(image, {rect.x, rect.y, std::max(rect.height, fill_width), rect.height}, color, std::max(2, rect.height / 2));
    }
}

void corner_box(cv::Mat& image, const cv::Rect& rect, cv::Scalar color, int thickness = 3) {
    const int length = std::max(12, std::min(rect.width, rect.height) / 4);
    const int x1 = rect.x;
    const int y1 = rect.y;
    const int x2 = rect.x + rect.width;
    const int y2 = rect.y + rect.height;
    for (const auto& [start, end] : {
             std::pair{cv::Point(x1, y1 + length), cv::Point(x1, y1)},
             std::pair{cv::Point(x1, y1), cv::Point(x1 + length, y1)},
             std::pair{cv::Point(x2 - length, y1), cv::Point(x2, y1)},
             std::pair{cv::Point(x2, y1), cv::Point(x2, y1 + length)},
             std::pair{cv::Point(x1, y2 - length), cv::Point(x1, y2)},
             std::pair{cv::Point(x1, y2), cv::Point(x1 + length, y2)},
             std::pair{cv::Point(x2 - length, y2), cv::Point(x2, y2)},
             std::pair{cv::Point(x2, y2 - length), cv::Point(x2, y2)},
         }) {
        cv::line(image, start, end, color, thickness, cv::LINE_AA);
    }
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

std::array<cv::Point, 21> landmark_pixels(const HandPrediction& hand, const cv::Size& size) {
    std::array<cv::Point, 21> points{};
    const int max_x = std::max(0, size.width - 1);
    const int max_y = std::max(0, size.height - 1);
    for (std::size_t i = 0; i < hand.landmarks.size(); ++i) {
        const int x = static_cast<int>(std::lround(hand.landmarks[i].x * size.width));
        const int y = static_cast<int>(std::lround(hand.landmarks[i].y * size.height));
        points[i] = {
            std::clamp(x, 0, max_x),
            std::clamp(y, 0, max_y),
        };
    }
    return points;
}

void draw_landmark_skeleton(
    cv::Mat& frame_bgr,
    const std::array<cv::Point, 21>& points,
    cv::Scalar color,
    bool estimated) {
    const int line_thickness = estimated ? 1 : 2;
    const int dot_radius = estimated ? 3 : 4;
    const cv::Scalar dot_fill = estimated ? cv::Scalar(80, 210, 255) : cv::Scalar(245, 245, 245);
    for (const auto& [start, end] : kHandConnections) {
        cv::line(frame_bgr, points[start], points[end], color, line_thickness, cv::LINE_AA);
    }
    for (const auto& point : points) {
        cv::circle(frame_bgr, point, dot_radius, dot_fill, -1, cv::LINE_AA);
        cv::circle(frame_bgr, point, dot_radius + 1, color, 1, cv::LINE_AA);
    }
}

void draw_status_row(cv::Mat& canvas, const cv::Rect& rect, const std::string& label, bool passed) {
    const cv::Scalar color = passed ? kSuccess : kAccent;
    rounded_rect(canvas, rect, kPanelAlt, 6, kBorder);
    cv::circle(canvas, {rect.x + 19, rect.y + rect.height / 2}, 9, color, -1, cv::LINE_AA);
    centered_text(canvas, passed ? "OK" : "--", {rect.x + 8, rect.y + 9, 22, 22}, 0.27, cv::Scalar(24, 24, 24), 1);
    put_text(canvas, label, {rect.x + 40, rect.y + 25}, 0.41, kText, 1);
    right_text(canvas, passed ? "PASS" : "WAIT", {rect.x + rect.width - 12, rect.y + 25}, 0.32, color, 1);
}

cv::Mat cover_image(const cv::Mat& source, const cv::Size& target) {
    cv::Mat resized;
    if (source.empty()) {
        return cv::Mat(target, CV_8UC3, kTopbar);
    }
    const double scale = std::max(
        static_cast<double>(target.width) / static_cast<double>(source.cols),
        static_cast<double>(target.height) / static_cast<double>(source.rows));
    const int width = std::max(1, static_cast<int>(std::lround(source.cols * scale)));
    const int height = std::max(1, static_cast<int>(std::lround(source.rows * scale)));
    cv::resize(source, resized, {width, height}, 0.0, 0.0, scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    const int crop_x = std::max(0, (width - target.width) / 2);
    const int crop_y = std::max(0, (height - target.height) / 2);
    return resized(cv::Rect(crop_x, crop_y, target.width, target.height)).clone();
}

std::string reason_title(const std::optional<CaptureGateDecision>& decision) {
    if (!decision) {
        return "CAPTURE GATE OFF";
    }
    if (decision->ready) {
        return "READY";
    }
    switch (decision->reason) {
        case GateReason::Ready:
            return "READY";
        case GateReason::GlassesPoseMissing:
            return "WAITING FOR POSE";
        case GateReason::GlassesPoseBad:
            return "ADJUST GLASSES";
        case GateReason::NeedTwoHands:
            return "SHOW BOTH HANDS";
        case GateReason::HandsOutOfFrame:
            return "HANDS OUT OF FRAME";
        case GateReason::HandsNotCentered:
            return "MOVE TO CENTER";
        case GateReason::HandsTooClose:
            return "SEPARATE HANDS";
        case GateReason::NeedDoubleOK:
            return "MAKE DOUBLE OK";
        case GateReason::AvoidDoubleOK:
            return "AVOID DOUBLE OK";
    }
    return "MONITORING";
}

std::string short_camera_label(const std::string& camera_label) {
    const auto pos = camera_label.find_first_of(" \t");
    return pos == std::string::npos ? camera_label : camera_label.substr(0, pos);
}

void draw_section_title(cv::Mat& image, const std::string& title, cv::Point origin) {
    cv::rectangle(image, {origin.x, origin.y - 13}, {origin.x + 4, origin.y + 2}, kAccent, -1);
    put_text(image, title, {origin.x + 13, origin.y}, 0.43, kMuted, 1);
}

}  // namespace

void draw_hand_tracking(cv::Mat& frame_bgr, const DoubleOKResult& result) {
    const cv::Size size = frame_bgr.size();
    for (const auto& hand : result.hands) {
        const cv::Scalar color = ok_color(hand.is_ok);
        const std::array<cv::Point, 21> points = landmark_pixels(hand, size);
        if (hand.landmarks_estimated) {
            const cv::Rect box = hand_box(hand, size);
            draw_landmark_skeleton(frame_bgr, points, cv::Scalar(80, 210, 255), true);
            corner_box(frame_bgr, box, color, 3);
            const std::string label = hand.handedness + " candidate " + fixed(hand.ok_score * 100.0, 1) + "%";
            const int label_width = std::max(190, cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, nullptr).width + 24);
            const int label_y = std::max(8, box.y - 34);
            rounded_rect(frame_bgr, {box.x, label_y, label_width, 28}, cv::Scalar(32, 32, 32), 6);
            put_text(
                frame_bgr,
                label,
                {box.x + 10, label_y + 19},
                0.48,
                color,
                1);
            put_text(
                frame_bgr,
                "OpenCV heuristic, not MediaPipe landmarks",
                {box.x, std::min(size.height - 10, box.y + box.height + 22)},
                0.42,
                cv::Scalar(80, 210, 255),
                1);
            continue;
        }
        draw_landmark_skeleton(frame_bgr, points, color, false);
        const cv::Rect box = hand_box(hand, size);
        corner_box(frame_bgr, box, color, 3);
        const std::string label = hand.handedness + "  " + fixed(hand.ok_score * 100.0, 1) + "%";
        const int label_width = std::max(150, cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, nullptr).width + 24);
        const int label_y = std::max(8, box.y - 34);
        rounded_rect(frame_bgr, {box.x, label_y, label_width, 28}, cv::Scalar(32, 32, 32), 6);
        put_text(
            frame_bgr,
            label,
            {box.x + 10, label_y + 19},
            0.48,
            color,
            1);
    }
}

void draw_capture_guides(cv::Mat& frame_bgr, const CaptureGateDecision& decision, const CaptureGateConfig& config) {
    const cv::Scalar color = decision.ready ? kSuccess : kAccent;
    const int x1 = static_cast<int>(config.center_x_min * frame_bgr.cols);
    const int x2 = static_cast<int>(config.center_x_max * frame_bgr.cols);
    const int y1 = static_cast<int>(config.center_y_min * frame_bgr.rows);
    const int y2 = static_cast<int>(config.center_y_max * frame_bgr.rows);
    corner_box(frame_bgr, {x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1)}, color, 3);
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
    const bool ready = decision && decision->ready;
    const cv::Scalar state_color = ready ? kSuccess : kAccent;
    cv::Mat canvas(height, width, CV_8UC3, kBackground);

    const int margin = std::max(16, width / 90);
    const int header_h = std::max(64, height / 12);
    const int sidebar_w = std::max(330, static_cast<int>(std::lround(static_cast<double>(width) * 0.245)));
    const int content_y = header_h + margin;
    const int content_h = height - content_y - margin;
    const int camera_w = width - sidebar_w - margin * 3;
    const int camera_x = margin;
    const int sidebar_x = camera_x + camera_w + margin;

    cv::rectangle(canvas, {0, 0}, {width, header_h}, kTopbar, -1);
    cv::line(canvas, {margin, 0}, {margin, header_h}, kAccent, 2, cv::LINE_AA);
    cv::circle(canvas, {margin, 14}, 5, kAccent, -1, cv::LINE_AA);
    put_text(canvas, "DOUBLE OK CAPTURE GATE", {margin + 20, 39}, 0.76, kText, 2);

    const cv::Rect status_rect(width - margin - 190, std::max(8, (header_h - 36) / 2), 190, 36);
    const int chip_w = 122;
    const int chip_gap = 8;
    const int leftmost_chip_x = status_rect.x - chip_gap - chip_w * 3 - chip_gap * 2;
    const int branch_x = margin + 280;
    const int branch_y = std::max(9, (header_h - 34) / 2);
    if (branch_x + 136 < leftmost_chip_x) {
        rounded_rect(canvas, {branch_x, branch_y, 86, 34}, kPanelAlt, 6, kBorder);
        cv::circle(canvas, {branch_x + 16, branch_y + 17}, 4, kAccent, -1, cv::LINE_AA);
        cv::line(canvas, {branch_x + 16, branch_y + 17}, {branch_x + 16, branch_y + 10}, kAccent, 1, cv::LINE_AA);
        cv::circle(canvas, {branch_x + 23, branch_y + 10}, 3, kAccent, 1, cv::LINE_AA);
        put_text(canvas, "main", {branch_x + 34, branch_y + 22}, 0.47, kText, 1);
        put_text(canvas, "03", {branch_x + 96, branch_y + 23}, 0.56, kText, 2);
    }

    rounded_rect(canvas, status_rect, state_color, 7);
    centered_text(canvas, reason_title(decision), status_rect, 0.42, cv::Scalar(24, 24, 24), 1);

    const std::array<std::tuple<std::string, std::string, bool>, 3> chips = {
        std::tuple{"FPS", fixed(snapshot.fps, 1), snapshot.fps >= target_fps || snapshot.frame_count < 10 || target_fps <= 0.0},
        std::tuple{"LATENCY", fixed(snapshot.processing_ms, 1) + " ms", snapshot.processing_ms <= 100.0},
        std::tuple{"CAMERA", short_camera_label(camera_label), true},
    };
    int chip_x = status_rect.x - chip_gap;
    for (auto it = chips.rbegin(); it != chips.rend(); ++it) {
        chip_x -= chip_w;
        rounded_rect(canvas, {chip_x, status_rect.y, chip_w, status_rect.height}, kPanel, 6, kBorder);
        put_text(canvas, std::get<0>(*it), {chip_x + 10, status_rect.y + 14}, 0.29, kMuted, 1);
        put_text(canvas, std::get<1>(*it), {chip_x + 10, status_rect.y + 30}, 0.38, std::get<2>(*it) ? kText : kWarning, 1);
        chip_x -= chip_gap;
    }
    cv::line(canvas, {0, header_h}, {width, header_h}, kBorder, 1);

    const cv::Rect camera_rect(camera_x, content_y, camera_w, content_h);
    rounded_rect(canvas, camera_rect, kPanel, 8, kBorder);
    const cv::Rect inner(camera_rect.x + 6, camera_rect.y + 6, camera_rect.width - 12, camera_rect.height - 12);
    cv::Mat fitted = cover_image(camera_frame, inner.size());
    fitted.copyTo(canvas(inner));
    cv::Mat shade = canvas(inner).clone();
    cv::rectangle(shade, {0, 0}, {inner.width, 62}, kTopbar, -1);
    cv::addWeighted(shade, 0.40, canvas(inner), 0.60, 0.0, canvas(inner));
    put_text(canvas, "LIVE CAMERA", {inner.x + 18, inner.y + 29}, 0.52, kText, 2);
    put_text(canvas, "TRACKING  /  TWO HANDS", {inner.x + 18, inner.y + 50}, 0.34, kMuted, 1);
    cv::circle(canvas, {inner.x + inner.width - 28, inner.y + 28}, 6, kDanger, -1, cv::LINE_AA);
    put_text(canvas, "LIVE", {inner.x + inner.width - 78, inner.y + 34}, 0.38, kText, 1);

    const int banner_h = 70;
    const int banner_y = inner.y + inner.height - banner_h;
    cv::Mat overlay = canvas(cv::Rect(inner.x, banner_y, inner.width, banner_h)).clone();
    cv::rectangle(overlay, {0, 0}, {inner.width, banner_h}, kTopbar, -1);
    cv::addWeighted(overlay, 0.86, canvas(cv::Rect(inner.x, banner_y, inner.width, banner_h)), 0.14, 0.0,
                    canvas(cv::Rect(inner.x, banner_y, inner.width, banner_h)));
    cv::rectangle(canvas, {inner.x, banner_y}, {inner.x + 6, inner.y + inner.height}, state_color, -1);
    put_text(canvas, reason_title(decision), {inner.x + 24, banner_y + 29}, 0.58, state_color, 2);
    put_text(
        canvas,
        decision ? decision->prompt : "Capture gate is disabled",
        {inner.x + 24, banner_y + 55},
        0.38,
        kText,
        1);

    const cv::Rect side(sidebar_x, content_y, sidebar_w, content_h);
    rounded_rect(canvas, side, kPanel, 8, kBorder);
    const int pad = 18;
    const int sx = side.x + pad;
    const int sw = side.width - pad * 2;
    int y = side.y + 28;
    draw_section_title(canvas, "CAPTURE READINESS", {sx, y});
    y += 24;
    const int passed = decision
                           ? static_cast<int>(decision->glasses_pose_ok) + static_cast<int>(decision->hands_visible) +
                                 static_cast<int>(decision->hands_centered) + static_cast<int>(decision->hands_separated) +
                                 static_cast<int>(decision->gesture_ok)
                           : 0;
    const int percent = decision ? passed * 20 : 0;
    put_text(canvas, std::to_string(percent) + "%", {sx, y + 28}, 0.78, kText, 2);
    progress_bar(canvas, {sx + 74, y + 11, sw - 74, 14}, static_cast<double>(percent) / 100.0, state_color, kPanelAlt);
    y += 54;
    if (decision) {
        for (const auto& [label, value] : {
                 std::pair{"POSE", decision->glasses_pose_ok},
                 std::pair{"VISIBLE", decision->hands_visible},
                 std::pair{"CENTERED", decision->hands_centered},
                 std::pair{"SEPARATED", decision->hands_separated},
                 std::pair{"GESTURE", decision->gesture_ok},
             }) {
            draw_status_row(canvas, {sx, y, sw, 40}, label, value);
            y += 46;
        }
    } else {
        put_text(canvas, "Capture gate disabled", {sx, y + 24}, 0.48, kMuted, 1);
        y += 230;
    }

    y += 12;
    draw_section_title(canvas, "HAND CONFIDENCE", {sx, y});
    y += 20;
    for (std::size_t i = 0; i < 2; ++i) {
        const cv::Rect card(sx, y, sw, 78);
        rounded_rect(canvas, card, kPanelAlt, 6, kBorder);
        if (i < result.hands.size()) {
            const auto& hand = result.hands[i];
            const cv::Scalar color = ok_color(hand.is_ok);
            put_text(canvas, hand.handedness, {card.x + 14, card.y + 24}, 0.43, kText, 1);
            right_text(canvas, fixed(hand.ok_score * 100.0, 1) + "%", {card.x + card.width - 14, card.y + 24}, 0.45, color, 1);
            progress_bar(canvas, {card.x + 14, card.y + 36, card.width - 28, 10}, hand.ok_score, color, kBackground);
            put_text(canvas, hand.is_ok ? "OK GESTURE" : "NOT OK", {card.x + 14, card.y + 65}, 0.32, color, 1);
        } else {
            put_text(canvas, "HAND " + std::to_string(i + 1), {card.x + 14, card.y + 24}, 0.43, kText, 1);
            put_text(canvas, "NOT DETECTED", {card.x + 14, card.y + 65}, 0.32, kMuted, 1);
        }
        y += 88;
    }

    if (side.y + side.height - y >= 115) {
        draw_section_title(canvas, "SYSTEM", {sx, y});
        y += 22;
        const std::array<std::pair<std::string, std::string>, 4> system_rows = {
            std::pair<std::string, std::string>{"CAMERA", camera_label},
            std::pair<std::string, std::string>{"MODEL", model_label},
            std::pair<std::string, std::string>{"FRAME", "#" + std::to_string(snapshot.frame_count)},
            std::pair<std::string, std::string>{"CONTROLS", "Q / ESC Exit    S Snapshot"},
        };
        for (const auto& [label, value] : system_rows) {
            put_text(canvas, label, {sx, y + 15}, 0.31, kMuted, 1);
            right_text(canvas, value, {sx + sw, y + 15}, 0.34, kText, 1);
            y += 25;
        }
    }
    cv::line(canvas, {sx, side.y + side.height - 42}, {sx + sw, side.y + side.height - 42}, kBorder, 1);
    put_text(canvas, "DOUBLE OK  /  V0.1", {sx, side.y + side.height - 18}, 0.32, kMuted, 1);
    right_text(canvas, "ONLINE", {sx + sw, side.y + side.height - 18}, 0.34, kSuccess, 1);
    cv::rectangle(canvas, {0, height - 2}, {width, height}, kAccent, -1);
    return canvas;
}

}  // namespace double_ok_gesture
