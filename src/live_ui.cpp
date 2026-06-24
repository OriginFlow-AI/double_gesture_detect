#include "double_ok_gesture/live_ui.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <tuple>

#ifdef DOUBLE_OK_HAS_OPENCV_FREETYPE
#include <opencv2/freetype.hpp>
#endif

namespace double_ok_gesture {
namespace {

const std::array<std::pair<int, int>, 21> kHandConnections = {
    std::pair{0, 1},   std::pair{1, 2},   std::pair{2, 3},   std::pair{3, 4},   std::pair{0, 5},
    std::pair{5, 6},   std::pair{6, 7},   std::pair{7, 8},   std::pair{5, 9},   std::pair{9, 10},
    std::pair{10, 11}, std::pair{11, 12}, std::pair{9, 13},  std::pair{13, 14}, std::pair{14, 15},
    std::pair{15, 16}, std::pair{13, 17}, std::pair{17, 18}, std::pair{18, 19}, std::pair{19, 20},
    std::pair{0, 17},
};

const cv::Scalar kBackground(22, 24, 28);
const cv::Scalar kTopbar(25, 27, 32);
const cv::Scalar kPanel(32, 35, 41);
const cv::Scalar kPanelSoft(36, 39, 46);
const cv::Scalar kBorder(72, 78, 90);
const cv::Scalar kSubtleBorder(50, 55, 64);
const cv::Scalar kOverlay(18, 20, 25);
const cv::Scalar kText(242, 245, 249);
const cv::Scalar kMuted(158, 166, 178);
const cv::Scalar kAccent(245, 139, 76);
const cv::Scalar kBrand(82, 145, 247);
const cv::Scalar kSuccess(92, 205, 128);
const cv::Scalar kWarning(92, 160, 255);
const cv::Scalar kDanger(80, 83, 239);

cv::Scalar ok_color(bool ok) {
    return ok ? kSuccess : kAccent;
}

std::string fixed(double value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

int font_height(double scale) {
    return std::max(9, static_cast<int>(std::lround(scale * 32.0)));
}

#ifdef DOUBLE_OK_HAS_OPENCV_FREETYPE
cv::Ptr<cv::freetype::FreeType2> load_font(const std::array<std::filesystem::path, 5>& candidates) {
    for (const auto& path : candidates) {
        if (!std::filesystem::exists(path)) {
            continue;
        }
        try {
            auto ft = cv::freetype::createFreeType2();
            ft->loadFontData(path.string(), 0);
            ft->setSplitNumber(8);
            return ft;
        } catch (const cv::Exception&) {
        }
    }
    return {};
}

cv::Ptr<cv::freetype::FreeType2> dashboard_font(bool bold = false) {
    static cv::Ptr<cv::freetype::FreeType2> regular = []() {
        const std::array<std::filesystem::path, 5> candidates = {
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        };
        return load_font(candidates);
    }();
    static cv::Ptr<cv::freetype::FreeType2> emphasis = []() {
        const std::array<std::filesystem::path, 5> candidates = {
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        };
        return load_font(candidates);
    }();
    return bold && emphasis ? emphasis : regular;
}
#endif

cv::Size text_size(const std::string& text, double scale, int thickness = 1, int* baseline = nullptr) {
#ifdef DOUBLE_OK_HAS_OPENCV_FREETYPE
    if (auto font = dashboard_font(thickness >= 2)) {
        int local_baseline = 0;
        const cv::Size size = font->getTextSize(text, font_height(scale), -1, &local_baseline);
        if (baseline) {
            *baseline = local_baseline;
        }
        return size;
    }
#endif
    return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, baseline);
}

bool prefers_localized_text() {
#ifdef DOUBLE_OK_HAS_OPENCV_FREETYPE
    return static_cast<bool>(dashboard_font(false));
#else
    return false;
#endif
}

std::string ui_text(const std::string& localized, const std::string& fallback) {
    return prefers_localized_text() ? localized : fallback;
}

void put_text(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar color, int thickness = 1) {
#ifdef DOUBLE_OK_HAS_OPENCV_FREETYPE
    if (auto font = dashboard_font(thickness >= 2)) {
        font->putText(image, text, origin, font_height(scale), color, -1, cv::LINE_AA, true);
        return;
    }
#endif
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

std::string fit_text(const std::string& text, int max_width, double scale, int thickness = 1) {
    if (max_width <= 0) {
        return "";
    }
    if (text_size(text, scale, thickness).width <= max_width) {
        return text;
    }
    const std::string suffix = "...";
    if (text_size(suffix, scale, thickness).width > max_width) {
        return "";
    }
    std::string clipped = text;
    while (!clipped.empty()) {
        clipped.pop_back();
        const std::string candidate = clipped + suffix;
        if (text_size(candidate, scale, thickness).width <= max_width) {
            return candidate;
        }
    }
    return suffix;
}

void blend_rect(cv::Mat& image, const cv::Rect& rect, cv::Scalar color, double alpha) {
    const cv::Rect clipped = rect & cv::Rect(0, 0, image.cols, image.rows);
    if (clipped.width <= 0 || clipped.height <= 0) {
        return;
    }
    alpha = std::clamp(alpha, 0.0, 1.0);
    cv::Mat roi = image(clipped);
    cv::Mat overlay(roi.size(), roi.type(), color);
    cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
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

void panel_rect(cv::Mat& image, const cv::Rect& rect, cv::Scalar color, int radius, std::optional<cv::Scalar> border = kBorder) {
    const cv::Rect shadow(rect.x + 2, rect.y + 3, rect.width, rect.height);
    rounded_rect(image, shadow & cv::Rect(0, 0, image.cols, image.rows), cv::Scalar(16, 17, 20), radius + 1);
    rounded_rect(image, rect, color, radius, border);
}

void right_text(cv::Mat& image, const std::string& text, cv::Point anchor, double scale, cv::Scalar color, int thickness = 1) {
    const int width = text_size(text, scale, thickness).width;
    put_text(image, text, {anchor.x - width, anchor.y}, scale, color, thickness);
}

void centered_text(cv::Mat& image, const std::string& text, const cv::Rect& rect, double scale, cv::Scalar color, int thickness = 1) {
    int baseline = 0;
    const cv::Size size = text_size(text, scale, thickness, &baseline);
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

void draw_camera_grid(cv::Mat& image, const cv::Rect& rect, cv::Scalar state_color) {
    const cv::Rect clipped = rect & cv::Rect(0, 0, image.cols, image.rows);
    if (clipped.width <= 0 || clipped.height <= 0) {
        return;
    }
    cv::Mat roi = image(clipped);
    cv::Mat overlay = roi.clone();
    const cv::Scalar grid_color(66, 72, 84);
    for (int i = 1; i < 3; ++i) {
        const int x = i * clipped.width / 3;
        cv::line(overlay, {x, 0}, {x, clipped.height}, grid_color, 1, cv::LINE_AA);
    }
    for (int i = 1; i < 3; ++i) {
        const int y = i * clipped.height / 3;
        cv::line(overlay, {0, y}, {clipped.width, y}, grid_color, 1, cv::LINE_AA);
    }
    cv::addWeighted(overlay, 0.16, roi, 0.84, 0.0, roi);

    const cv::Point center(clipped.x + clipped.width / 2, clipped.y + clipped.height / 2);
    const int tick = std::max(18, std::min(clipped.width, clipped.height) / 18);
    cv::line(image, {center.x - tick, center.y}, {center.x - 6, center.y}, state_color, 1, cv::LINE_AA);
    cv::line(image, {center.x + 6, center.y}, {center.x + tick, center.y}, state_color, 1, cv::LINE_AA);
    cv::line(image, {center.x, center.y - tick}, {center.x, center.y - 6}, state_color, 1, cv::LINE_AA);
    cv::line(image, {center.x, center.y + 6}, {center.x, center.y + tick}, state_color, 1, cv::LINE_AA);
    cv::circle(image, center, 3, state_color, -1, cv::LINE_AA);
}

void draw_metric_pill(
    cv::Mat& image,
    const cv::Rect& rect,
    const std::string& label,
    const std::string& value,
    cv::Scalar color) {
    rounded_rect(image, rect, kOverlay, std::max(6, rect.height / 2), kSubtleBorder);
    cv::circle(image, {rect.x + 14, rect.y + rect.height / 2}, 4, color, -1, cv::LINE_AA);
    put_text(image, label, {rect.x + 26, rect.y + 15}, 0.27, kMuted, 1);
    put_text(image, fit_text(value, rect.width - 34, 0.35), {rect.x + 26, rect.y + 31}, 0.35, kText, 1);
}

void draw_center_notice(
    cv::Mat& image,
    const cv::Rect& rect,
    const std::string& title,
    const std::string& subtitle,
    cv::Scalar color) {
    if (rect.width < 180 || rect.height < 180) {
        return;
    }
    const int box_w = std::min(std::max(330, rect.width / 2), rect.width - 48);
    const int box_h = 96;
    const cv::Rect box(rect.x + (rect.width - box_w) / 2, rect.y + (rect.height - box_h) / 2, box_w, box_h);
    blend_rect(image, box, kOverlay, 0.76);
    rounded_rect(image, box, kOverlay, 8, kSubtleBorder);
    cv::circle(image, {box.x + 34, box.y + 38}, 14, color, -1, cv::LINE_AA);
    centered_text(image, "!", {box.x + 20, box.y + 24, 28, 28}, 0.40, cv::Scalar(24, 24, 24), 1);
    put_text(image, fit_text(title, box.width - 78, 0.52, 2), {box.x + 64, box.y + 38}, 0.52, kText, 2);
    put_text(image, fit_text(subtitle, box.width - 78, 0.36), {box.x + 64, box.y + 64}, 0.36, kMuted, 1);
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

void draw_status_row(
    cv::Mat& canvas,
    const cv::Rect& rect,
    const std::string& label,
    bool passed) {
    const cv::Scalar color = passed ? kSuccess : kAccent;
    const int cy = rect.y + rect.height / 2;
    rounded_rect(canvas, rect, kPanelSoft, 6, std::nullopt);
    cv::line(canvas, {rect.x + 12, rect.y + rect.height}, {rect.x + rect.width - 12, rect.y + rect.height}, kSubtleBorder, 1, cv::LINE_AA);

    cv::circle(canvas, {rect.x + 21, cy}, 11, color, -1, cv::LINE_AA);
    centered_text(canvas, passed ? ui_text("过", "OK") : "--", {rect.x + 10, cy - 11, 22, 22}, 0.25, cv::Scalar(24, 24, 24), 1);

    const int text_x = rect.x + 46;
    put_text(canvas, label, {text_x, cy + 5}, 0.39, kText, 1);
    right_text(
        canvas,
        passed ? ui_text("通过", "PASS") : ui_text("等待", "WAIT"),
        {rect.x + rect.width - 14, cy + 5},
        0.31,
        color,
        1);
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

bool is_dark_frame(const cv::Mat& image) {
    if (image.empty()) {
        return false;
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray, mean, stddev);
    return mean[0] < 12.0 && stddev[0] < 28.0;
}

bool is_unsupported_camera_stream(const std::string& camera_label) {
    return camera_label.find("NV12") != std::string::npos || camera_label.find("Z16") != std::string::npos ||
           camera_label.find("GREY") != std::string::npos;
}

std::string reason_title(const std::optional<CaptureGateDecision>& decision) {
    if (!decision) {
        return ui_text("门控关闭", "CAPTURE GATE OFF");
    }
    if (decision->ready) {
        return ui_text("已就绪", "READY");
    }
    switch (decision->reason) {
        case GateReason::Ready:
            return ui_text("已就绪", "READY");
        case GateReason::GlassesPoseMissing:
            return ui_text("等待姿态", "WAITING FOR POSE");
        case GateReason::GlassesPoseBad:
            return ui_text("调整姿态", "ADJUST GLASSES");
        case GateReason::NeedTwoHands:
            return ui_text("请露出双手", "SHOW BOTH HANDS");
        case GateReason::HandsOutOfFrame:
            return ui_text("双手未完整入框", "HANDS OUT OF FRAME");
        case GateReason::HandsNotCentered:
            return ui_text("移动到中心", "MOVE TO CENTER");
        case GateReason::HandsTooClose:
            return ui_text("拉开双手间距", "SEPARATE HANDS");
        case GateReason::NeedDoubleOK:
            return ui_text("做出双手 OK", "MAKE DOUBLE OK");
        case GateReason::AvoidDoubleOK:
            return ui_text("避免双手 OK", "AVOID DOUBLE OK");
    }
    return ui_text("监测中", "MONITORING");
}

std::string reason_hint(const std::optional<CaptureGateDecision>& decision) {
    if (!decision) {
        return ui_text("采集门控已关闭，仅显示追踪结果。", "Capture gate disabled. Tracking overlay only.");
    }
    if (decision->ready) {
        return ui_text("条件已满足，可以开始采集。", "Conditions passed. Capture can start.");
    }
    switch (decision->reason) {
        case GateReason::Ready:
            return ui_text("条件已满足，可以开始采集。", "Conditions passed. Capture can start.");
        case GateReason::GlassesPoseMissing:
            return ui_text("等待眼镜姿态数据。", "Waiting for glasses pose data.");
        case GateReason::GlassesPoseBad:
            return ui_text("请调整眼镜角度，让视野正对双手。", "Adjust glasses angle toward both hands.");
        case GateReason::NeedTwoHands:
            return ui_text("请把双手放入相机画面。", "Place both hands inside the camera view.");
        case GateReason::HandsOutOfFrame:
            return ui_text("请让双手完整进入画面。", "Keep both hands fully inside the frame.");
        case GateReason::HandsNotCentered:
            return ui_text("请把双手移动到中心框内。", "Move both hands into the center guide.");
        case GateReason::HandsTooClose:
            return ui_text("请把两只手再分开一些。", "Separate your hands a little more.");
        case GateReason::NeedDoubleOK:
            return ui_text("请双手分开，并做出清晰 OK 手势。", "Make two clear OK gestures with separated hands.");
        case GateReason::AvoidDoubleOK:
            return ui_text("负样本采集中，请避免同时做双手 OK。", "Negative capture: avoid double OK gestures.");
    }
    return ui_text("正在监测采集条件。", "Monitoring capture conditions.");
}

std::string short_camera_label(const std::string& camera_label) {
    const auto pos = camera_label.find_first_of(" \t");
    return pos == std::string::npos ? camera_label : camera_label.substr(0, pos);
}

std::string handedness_label(const std::string& handedness) {
    if (!prefers_localized_text()) {
        return handedness;
    }
    if (handedness == "Left") {
        return "左手";
    }
    if (handedness == "Right") {
        return "右手";
    }
    return "未知手";
}

void draw_section_title(cv::Mat& image, const std::string& title, cv::Point origin) {
    cv::rectangle(image, {origin.x, origin.y - 15}, {origin.x + 3, origin.y + 3}, kAccent, -1);
    put_text(image, title, {origin.x + 12, origin.y}, 0.38, kMuted, 1);
}

}  // namespace

void draw_hand_tracking(cv::Mat& frame_bgr, const DoubleOKResult& result) {
    const cv::Size size = frame_bgr.size();
    for (const auto& hand : result.hands) {
        const cv::Scalar color = ok_color(hand.is_ok);
        if (hand.landmarks_estimated) {
            const cv::Rect box = hand_box(hand, size);
            corner_box(frame_bgr, box, color, 3);
            const std::string label = handedness_label(hand.handedness) + ui_text(" 候选框 ", " box ") + fixed(hand.ok_score * 100.0, 1) + "%";
            const int label_width = std::max(178, text_size(label, 0.48).width + 24);
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
                ui_text("OpenCV 候选框，未输出 21 点关键点", "OpenCV heuristic box, no 21-point landmarks"),
                {box.x, std::min(size.height - 10, box.y + box.height + 22)},
                0.42,
                cv::Scalar(80, 210, 255),
                1);
            continue;
        }
        const std::array<cv::Point, 21> points = landmark_pixels(hand, size);
        draw_landmark_skeleton(frame_bgr, points, color, false);
        const cv::Rect box = hand_box(hand, size);
        corner_box(frame_bgr, box, color, 3);
        const std::string label = handedness_label(hand.handedness) + "  " + fixed(hand.ok_score * 100.0, 1) + "%";
        const int label_width = std::max(150, text_size(label, 0.48).width + 24);
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
    const int header_h = std::max(68, height / 11);
    const int sidebar_w = std::max(330, static_cast<int>(std::lround(static_cast<double>(width) * 0.245)));
    const int content_y = header_h + margin;
    const int content_h = height - content_y - margin;
    const int camera_w = width - sidebar_w - margin * 3;
    const int camera_x = margin;
    const int sidebar_x = camera_x + camera_w + margin;

    cv::rectangle(canvas, {0, 0}, {width, header_h}, kTopbar, -1);
    cv::line(canvas, {margin, 14}, {margin, header_h - 14}, kBrand, 3, cv::LINE_AA);
    cv::circle(canvas, {margin, 16}, 5, kBrand, -1, cv::LINE_AA);

    const cv::Rect status_rect(width - margin - 198, std::max(12, (header_h - 40) / 2), 198, 40);
    const int chip_w = 128;
    const int chip_gap = 10;
    const std::array<std::tuple<std::string, std::string, bool>, 3> chips = {
        std::tuple{ui_text("帧率", "FPS"), fixed(snapshot.fps, 1), snapshot.fps >= target_fps || snapshot.frame_count < 10 || target_fps <= 0.0},
        std::tuple{ui_text("延迟", "LATENCY"), fixed(snapshot.processing_ms, 1) + " ms", snapshot.processing_ms <= 100.0},
        std::tuple{ui_text("相机", "CAMERA"), short_camera_label(camera_label), true},
    };
    const int title_x = margin + 22;
    const int title_y = 43;
    const std::string title = ui_text("双手 OK 采集门控", "DOUBLE OK CAPTURE GATE");
    const int title_width = text_size(title, 0.60, 1).width;
    auto leftmost_chip_for = [&](int count) {
        return status_rect.x - chip_gap - chip_w * count - chip_gap * std::max(0, count - 1);
    };
    int visible_chip_count = static_cast<int>(chips.size());
    while (visible_chip_count > 1 && title_x + title_width + 18 > leftmost_chip_for(visible_chip_count)) {
        --visible_chip_count;
    }
    const int leftmost_chip_x = leftmost_chip_for(visible_chip_count);
    const int title_max_width = std::max(120, leftmost_chip_x - title_x - 18);
    put_text(canvas, fit_text(title, title_max_width, 0.60, 1), {title_x, title_y}, 0.60, kText, 1);

    rounded_rect(canvas, status_rect, state_color, 8);
    centered_text(canvas, reason_title(decision), status_rect, 0.42, cv::Scalar(24, 24, 24), 1);

    int chip_x = status_rect.x - chip_gap;
    for (int i = visible_chip_count - 1; i >= 0; --i) {
        const auto& chip = chips[static_cast<std::size_t>(i)];
        chip_x -= chip_w;
        rounded_rect(canvas, {chip_x, status_rect.y, chip_w, status_rect.height}, kPanelSoft, 7, kSubtleBorder);
        put_text(canvas, std::get<0>(chip), {chip_x + 10, status_rect.y + 14}, 0.29, kMuted, 1);
        put_text(canvas, fit_text(std::get<1>(chip), chip_w - 20, 0.38), {chip_x + 10, status_rect.y + 30}, 0.38, std::get<2>(chip) ? kText : kWarning, 1);
        chip_x -= chip_gap;
    }
    cv::line(canvas, {0, header_h}, {width, header_h}, kSubtleBorder, 1);

    const cv::Rect camera_rect(camera_x, content_y, camera_w, content_h);
    panel_rect(canvas, camera_rect, kPanel, 8, kBorder);
    const cv::Rect inner(camera_rect.x + 8, camera_rect.y + 8, camera_rect.width - 16, camera_rect.height - 16);
    const bool has_camera_frame = !camera_frame.empty();
    const bool dark_camera_frame = has_camera_frame && is_dark_frame(camera_frame);
    const bool unsupported_camera_stream = is_unsupported_camera_stream(camera_label);
    cv::Mat fitted = cover_image(camera_frame, inner.size());
    fitted.copyTo(canvas(inner));
    draw_camera_grid(canvas, inner, state_color);
    if (!has_camera_frame) {
        draw_center_notice(canvas, inner, ui_text("无相机画面", "NO CAMERA FRAME"), ui_text("正在等待下一帧视频。", "Waiting for the next video frame."), kWarning);
    } else if (unsupported_camera_stream) {
        draw_center_notice(
            canvas,
            inner,
            ui_text("非彩色视频流", "UNSUPPORTED STREAM"),
            ui_text("请切换到 MJPG/YUYV 彩色视频节点。", "Use an MJPG/YUYV color video node."),
            kWarning);
    } else if (dark_camera_frame) {
        draw_center_notice(canvas, inner, ui_text("画面过暗", "DARK CAMERA FRAME"), ui_text("请检查相机源或曝光。", "Check camera source or exposure."), kWarning);
    } else if (result.hands.empty()) {
        draw_center_notice(canvas, inner, ui_text("等待双手", "WAITING FOR HANDS"), ui_text("请将双手放入中心框内。", "Place both hands inside the center guide."), state_color);
    }
    blend_rect(canvas, {inner.x, inner.y, inner.width, 66}, kTopbar, 0.58);
    cv::line(canvas, {inner.x, inner.y + 66}, {inner.x + inner.width, inner.y + 66}, kSubtleBorder, 1, cv::LINE_AA);
    cv::rectangle(canvas, {inner.x, inner.y}, {inner.x + 5, inner.y + 66}, state_color, -1);
    put_text(canvas, ui_text("实时画面", "LIVE CAMERA"), {inner.x + 20, inner.y + 30}, 0.54, kText, 2);
    put_text(
        canvas,
        !has_camera_frame
            ? ui_text("等待视频帧", "WAITING FOR FRAME")
            : (unsupported_camera_stream
                   ? ui_text("非彩色视频流", "UNSUPPORTED STREAM")
                   : (dark_camera_frame ? ui_text("检查相机源", "CHECK CAMERA SOURCE") : (result.stable_double_ok ? ui_text("双手 OK 已稳定", "STABLE DOUBLE OK") : ui_text("追踪 / 双手", "TRACKING  /  TWO HANDS")))),
        {inner.x + 20, inner.y + 52},
        0.34,
        result.stable_double_ok ? kSuccess : kMuted,
        1);
    const int live_pill_w = 82;
    const int hand_pill_w = 128;
    const int pill_h = 36;
    const int live_x = inner.x + inner.width - 18 - live_pill_w;
    const int hand_x = live_x - 10 - hand_pill_w;
    if (hand_x > inner.x + 220) {
        draw_metric_pill(
            canvas,
            {hand_x, inner.y + 15, hand_pill_w, pill_h},
            ui_text("双手", "HANDS"),
            std::to_string(result.hands.size()) + " / " + std::to_string(result.ok_count) + " OK",
            state_color);
    }
    rounded_rect(canvas, {live_x, inner.y + 15, live_pill_w, pill_h}, kOverlay, 18, kSubtleBorder);
    cv::circle(canvas, {live_x + 18, inner.y + 33}, 6, has_camera_frame && !dark_camera_frame && !unsupported_camera_stream ? kDanger : kWarning, -1, cv::LINE_AA);
    put_text(canvas, has_camera_frame && !dark_camera_frame && !unsupported_camera_stream ? ui_text("实时", "LIVE") : ui_text("等待", "WAIT"), {live_x + 34, inner.y + 38}, 0.37, kText, 1);

    const int banner_h = 82;
    const int banner_y = inner.y + inner.height - banner_h;
    blend_rect(canvas, {inner.x, banner_y, inner.width, banner_h}, kTopbar, 0.84);
    cv::line(canvas, {inner.x, banner_y}, {inner.x + inner.width, banner_y}, kSubtleBorder, 1, cv::LINE_AA);
    cv::rectangle(canvas, {inner.x, banner_y}, {inner.x + 7, inner.y + inner.height}, state_color, -1);
    cv::circle(canvas, {inner.x + 29, banner_y + 34}, 13, state_color, -1, cv::LINE_AA);
    centered_text(canvas, ready ? ui_text("采", "GO") : "!", {inner.x + 16, banner_y + 21, 26, 26}, 0.35, cv::Scalar(24, 24, 24), 1);
    const int banner_text_x = inner.x + 56;
    const int banner_text_w = inner.width - 78;
    put_text(canvas, fit_text(reason_title(decision), banner_text_w, 0.58, 2), {banner_text_x, banner_y + 31}, 0.58, state_color, 2);
    put_text(
        canvas,
        fit_text(reason_hint(decision), banner_text_w, 0.39),
        {banner_text_x, banner_y + 59},
        0.38,
        kText,
        1);

    const cv::Rect side(sidebar_x, content_y, sidebar_w, content_h);
    panel_rect(canvas, side, kPanel, 8, kBorder);
    const bool compact_side = side.height < 620;
    const int pad = compact_side ? 14 : 18;
    const int status_row_h = compact_side ? 34 : 40;
    const int status_row_gap = compact_side ? 38 : 46;
    const int hand_card_h = compact_side ? 62 : 78;
    const int hand_card_gap = compact_side ? 70 : 88;
    const int sx = side.x + pad;
    const int sw = side.width - pad * 2;
    int y = side.y + (compact_side ? 24 : 28);
    draw_section_title(canvas, ui_text("采集条件", "CAPTURE READINESS"), {sx, y});
    y += compact_side ? 22 : 24;
    const int passed = decision
                           ? static_cast<int>(decision->glasses_pose_ok) + static_cast<int>(decision->hands_visible) +
                                 static_cast<int>(decision->hands_centered) + static_cast<int>(decision->hands_separated) +
                                 static_cast<int>(decision->gesture_ok)
                           : 0;
    const int percent = decision ? passed * 20 : 0;
    put_text(canvas, std::to_string(percent) + "%", {sx, y + (compact_side ? 25 : 28)}, compact_side ? 0.68 : 0.78, kText, 2);
    progress_bar(
        canvas,
        {sx + 74, y + (compact_side ? 9 : 11), sw - 74, compact_side ? 12 : 14},
        static_cast<double>(percent) / 100.0,
        state_color,
        kPanelSoft);
    y += compact_side ? 44 : 54;
    if (decision) {
        for (const auto& [label, value] : {
                 std::pair{ui_text("姿态", "POSE"), decision->glasses_pose_ok},
                 std::pair{ui_text("双手可见", "VISIBLE"), decision->hands_visible},
                 std::pair{ui_text("居中", "CENTERED"), decision->hands_centered},
                 std::pair{ui_text("间距", "SPACING"), decision->hands_separated},
                 std::pair{ui_text("OK 手势", "GESTURE"), decision->gesture_ok},
             }) {
            draw_status_row(canvas, {sx, y, sw, status_row_h}, label, value);
            y += status_row_gap;
        }
    } else {
        put_text(canvas, ui_text("采集门控已关闭", "Capture gate disabled"), {sx, y + 24}, 0.48, kMuted, 1);
        y += compact_side ? 185 : 230;
    }

    y += compact_side ? 8 : 12;
    draw_section_title(canvas, ui_text("手部置信度", "HAND CONFIDENCE"), {sx, y});
    y += compact_side ? 18 : 20;
    for (std::size_t i = 0; i < 2; ++i) {
        const cv::Rect card(sx, y, sw, hand_card_h);
        rounded_rect(canvas, card, kPanelSoft, 7, kSubtleBorder);
        if (i < result.hands.size()) {
            const auto& hand = result.hands[i];
            const cv::Scalar color = ok_color(hand.is_ok);
            put_text(canvas, handedness_label(hand.handedness), {card.x + 14, card.y + 24}, 0.43, kText, 1);
            right_text(canvas, fixed(hand.ok_score * 100.0, 1) + "%", {card.x + card.width - 14, card.y + 24}, 0.45, color, 1);
            progress_bar(
                canvas,
                {card.x + 14, card.y + (compact_side ? 34 : 36), card.width - 28, compact_side ? 8 : 10},
                hand.ok_score,
                color,
                kBackground);
            put_text(
                canvas,
                hand.is_ok ? ui_text("OK 手势", "OK GESTURE") : ui_text("非 OK", "NOT OK"),
                {card.x + 14, card.y + (compact_side ? 54 : 65)},
                0.32,
                color,
                1);
        } else {
            put_text(canvas, ui_text("手 " + std::to_string(i + 1), "HAND " + std::to_string(i + 1)), {card.x + 14, card.y + 24}, 0.43, kText, 1);
            put_text(canvas, ui_text("未检测", "NOT DETECTED"), {card.x + 14, card.y + (compact_side ? 54 : 65)}, 0.32, kMuted, 1);
        }
        y += hand_card_gap;
    }

    if (side.y + side.height - y >= 115) {
        draw_section_title(canvas, ui_text("系统", "SYSTEM"), {sx, y});
        y += 22;
        const std::array<std::pair<std::string, std::string>, 4> system_rows = {
            std::pair<std::string, std::string>{ui_text("相机", "CAMERA"), camera_label},
            std::pair<std::string, std::string>{ui_text("模型", "MODEL"), model_label},
            std::pair<std::string, std::string>{ui_text("帧号", "FRAME"), "#" + std::to_string(snapshot.frame_count)},
            std::pair<std::string, std::string>{ui_text("控制", "CONTROLS"), ui_text("Q / ESC 退出    S 截图", "Q / ESC Exit    S Snapshot")},
        };
        for (const auto& [label, value] : system_rows) {
            put_text(canvas, label, {sx, y + 15}, 0.31, kMuted, 1);
            right_text(canvas, fit_text(value, sw - 86, 0.34), {sx + sw, y + 15}, 0.34, kText, 1);
            y += 25;
        }
    }
    cv::line(canvas, {sx, side.y + side.height - 42}, {sx + sw, side.y + side.height - 42}, kSubtleBorder, 1);
    put_text(canvas, ui_text("双手 OK  /  V0.2", "DOUBLE OK  /  V0.2"), {sx, side.y + side.height - 18}, 0.32, kMuted, 1);
    right_text(canvas, ui_text("在线", "ONLINE"), {sx + sw, side.y + side.height - 18}, 0.34, kSuccess, 1);
    cv::rectangle(canvas, {0, height - 2}, {width, height}, kBrand, -1);
    return canvas;
}

}  // namespace double_ok_gesture
