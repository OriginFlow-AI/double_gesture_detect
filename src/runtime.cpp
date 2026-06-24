#include "double_ok_gesture/runtime.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace double_ok_gesture {

double monotonic_seconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

RuntimeMetrics::RuntimeMetrics(std::size_t window_size) : window_size_(window_size) {
    if (window_size < 2) {
        throw std::invalid_argument("window_size must be at least 2");
    }
}

RuntimeSnapshot RuntimeMetrics::update(double frame_started, double frame_finished) {
    if (!std::isfinite(frame_started) || !std::isfinite(frame_finished)) {
        throw std::invalid_argument("frame timestamps must be finite");
    }
    if (frame_finished < frame_started) {
        throw std::invalid_argument("frame_finished must not precede frame_started");
    }

    timestamps_.push_back(frame_finished);
    while (timestamps_.size() > window_size_) {
        timestamps_.pop_front();
    }
    ++frame_count_;

    double fps = 0.0;
    if (timestamps_.size() >= 2) {
        const double elapsed = timestamps_.back() - timestamps_.front();
        if (elapsed > 0.0) {
            fps = static_cast<double>(timestamps_.size() - 1) / elapsed;
        }
    }
    return {fps, (frame_finished - frame_started) * 1000.0, frame_count_};
}

RuntimeSnapshot RuntimeMetrics::update(double frame_started) {
    return update(frame_started, monotonic_seconds());
}

void configure_logging(const std::string& level) {
    if (level != "DEBUG" && level != "INFO" && level != "WARNING" && level != "ERROR") {
        throw std::invalid_argument("Unsupported log level: " + level);
    }
}

}  // namespace double_ok_gesture
