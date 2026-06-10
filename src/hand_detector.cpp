#include "double_ok_gesture/hand_detector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

struct Candidate {
    std::vector<cv::Point> contour;
    cv::Rect box;
    double area = 0.0;
    double ok_score = 0.0;
};

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

cv::Mat build_skin_mask(const cv::Mat& frame_bgr, double blur_size) {
    cv::Mat ycrcb;
    cv::cvtColor(frame_bgr, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::Mat mask_ycrcb;
    cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 180, 135), mask_ycrcb);

    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask_hsv;
    cv::inRange(hsv, cv::Scalar(0, 20, 30), cv::Scalar(25, 255, 255), mask_hsv);

    cv::Mat mask;
    cv::bitwise_and(mask_ycrcb, mask_hsv, mask);
    if (blur_size >= 3.0) {
        int k = static_cast<int>(std::lround(blur_size));
        if (k % 2 == 0) {
            ++k;
        }
        cv::medianBlur(mask, mask, k);
    }

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, {-1, -1}, 2);
    return mask;
}

double contour_solidity(const std::vector<cv::Point>& contour, double area) {
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    const double hull_area = std::abs(cv::contourArea(hull));
    if (hull_area < 1.0) {
        return 0.0;
    }
    return clamp01(area / hull_area);
}

double convexity_score(const std::vector<cv::Point>& contour, const cv::Rect& box) {
    if (contour.size() < 4) {
        return 0.0;
    }
    std::vector<int> hull_indices;
    cv::convexHull(contour, hull_indices, false, false);
    if (hull_indices.size() < 4) {
        return 0.0;
    }

    std::vector<cv::Vec4i> defects;
    try {
        cv::convexityDefects(contour, hull_indices, defects);
    } catch (const cv::Exception&) {
        return 0.0;
    }
    const double scale = std::max(1.0, static_cast<double>(std::max(box.width, box.height)));
    int deep_count = 0;
    double depth_sum = 0.0;
    for (const auto& defect : defects) {
        const double depth = static_cast<double>(defect[3]) / 256.0;
        if (depth / scale > 0.035) {
            ++deep_count;
            depth_sum += depth / scale;
        }
    }
    return clamp01(0.20 * deep_count + depth_sum);
}

double hole_score_for_contour(const cv::Mat& hand_mask, const cv::Rect& box, double hand_area) {
    if (box.width < 10 || box.height < 10) {
        return 0.0;
    }
    cv::Mat roi = hand_mask(box).clone();
    cv::Mat inverse;
    cv::bitwise_not(roi, inverse);
    cv::rectangle(inverse, {0, 0}, {inverse.cols - 1, inverse.rows - 1}, cv::Scalar(0), 2);

    std::vector<std::vector<cv::Point>> holes;
    cv::findContours(inverse, holes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double best = 0.0;
    for (const auto& hole : holes) {
        const double area = std::abs(cv::contourArea(hole));
        if (area < std::max(18.0, hand_area * 0.004) || area > hand_area * 0.18) {
            continue;
        }
        const double perimeter = cv::arcLength(hole, true);
        if (perimeter < 1.0) {
            continue;
        }
        const double circularity = clamp01(4.0 * M_PI * area / (perimeter * perimeter));
        const cv::Rect hole_box = cv::boundingRect(hole);
        const double aspect = static_cast<double>(std::min(hole_box.width, hole_box.height)) /
                              static_cast<double>(std::max(1, std::max(hole_box.width, hole_box.height)));
        best = std::max(best, clamp01(0.65 * circularity + 0.35 * aspect));
    }
    return best;
}

double ok_score_for_candidate(const std::vector<cv::Point>& contour, const cv::Rect& box, const cv::Mat& mask, double area) {
    const double solidity = contour_solidity(contour, area);
    const double hole = hole_score_for_contour(mask, box, area);
    const double concavity = convexity_score(contour, box);
    const double shape = 1.0 - std::abs(solidity - 0.78) / 0.35;
    return clamp01(0.10 + 0.58 * hole + 0.22 * concavity + 0.10 * clamp01(shape));
}

Point3 normalized_point(double x, double y, const cv::Size& size) {
    return {
        clamp01(x / static_cast<double>(std::max(1, size.width))),
        clamp01(y / static_cast<double>(std::max(1, size.height))),
        0.0,
    };
}

Landmarks approximate_landmarks(const cv::Rect& box, const cv::Size& size) {
    const double x = static_cast<double>(box.x);
    const double y = static_cast<double>(box.y);
    const double w = static_cast<double>(box.width);
    const double h = static_cast<double>(box.height);
    const std::array<cv::Point2d, 21> template_points = {
        cv::Point2d(0.50, 0.96),
        cv::Point2d(0.30, 0.78),
        cv::Point2d(0.20, 0.62),
        cv::Point2d(0.16, 0.48),
        cv::Point2d(0.16, 0.34),
        cv::Point2d(0.40, 0.72),
        cv::Point2d(0.36, 0.52),
        cv::Point2d(0.34, 0.32),
        cv::Point2d(0.34, 0.12),
        cv::Point2d(0.52, 0.70),
        cv::Point2d(0.52, 0.46),
        cv::Point2d(0.52, 0.24),
        cv::Point2d(0.52, 0.04),
        cv::Point2d(0.64, 0.74),
        cv::Point2d(0.68, 0.52),
        cv::Point2d(0.70, 0.34),
        cv::Point2d(0.72, 0.16),
        cv::Point2d(0.76, 0.82),
        cv::Point2d(0.84, 0.64),
        cv::Point2d(0.88, 0.48),
        cv::Point2d(0.92, 0.32),
    };

    Landmarks landmarks{};
    for (std::size_t i = 0; i < template_points.size(); ++i) {
        landmarks[i] = normalized_point(x + template_points[i].x * w, y + template_points[i].y * h, size);
    }
    return landmarks;
}

}  // namespace

OpenCVHandDetector::OpenCVHandDetector(HandDetectorConfig config) : config_(config) {
    if (config_.max_num_hands < 1) {
        throw std::invalid_argument("max_num_hands must be positive");
    }
    if (config_.min_area_ratio <= 0.0 || config_.min_area_ratio >= 1.0) {
        throw std::invalid_argument("min_area_ratio must be in (0, 1)");
    }
}

std::vector<DetectedHand> OpenCVHandDetector::detect(const cv::Mat& frame_bgr) const {
    if (frame_bgr.empty() || frame_bgr.channels() != 3) {
        return {};
    }
    const cv::Mat mask = build_skin_mask(frame_bgr, config_.skin_mask_blur);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double frame_area = static_cast<double>(frame_bgr.cols * frame_bgr.rows);
    const double min_area = frame_area * config_.min_area_ratio;
    std::vector<Candidate> candidates;
    for (const auto& contour : contours) {
        const double area = std::abs(cv::contourArea(contour));
        if (area < min_area) {
            continue;
        }
        const cv::Rect box = cv::boundingRect(contour) & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
        if (box.width < 20 || box.height < 20) {
            continue;
        }
        candidates.push_back({contour, box, area, ok_score_for_candidate(contour, box, mask, area)});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.area > rhs.area;
    });
    if (candidates.size() > static_cast<std::size_t>(config_.max_num_hands)) {
        candidates.resize(static_cast<std::size_t>(config_.max_num_hands));
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.box.x < rhs.box.x;
    });

    std::vector<DetectedHand> hands;
    hands.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const std::string handedness = i == 0 && candidates.size() > 1 ? "Left" : (i == 1 ? "Right" : "Unknown");
        hands.push_back({
            approximate_landmarks(candidates[i].box, frame_bgr.size()),
            handedness,
            candidates[i].ok_score,
            true,
        });
    }
    return hands;
}

}  // namespace double_ok_gesture
