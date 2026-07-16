#include "double_ok_gesture/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

std::atomic<int> g_log_level{static_cast<int>(LogLevel::Info)};
std::mutex g_log_mutex;

const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warning:
            return "WARNING";
        case LogLevel::Error:
            return "ERROR";
    }
    return "UNKNOWN";
}

}  // namespace

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
    if (level == "DEBUG") {
        g_log_level.store(static_cast<int>(LogLevel::Debug));
    } else if (level == "INFO") {
        g_log_level.store(static_cast<int>(LogLevel::Info));
    } else if (level == "WARNING") {
        g_log_level.store(static_cast<int>(LogLevel::Warning));
    } else if (level == "ERROR") {
        g_log_level.store(static_cast<int>(LogLevel::Error));
    } else {
        throw std::invalid_argument(
            "Unsupported log level '" + level +
            "'; expected DEBUG, INFO, WARNING, or ERROR");
    }
}

void log_message(LogLevel level, std::string_view message) {
    if (static_cast<int>(level) < g_log_level.load()) {
        return;
    }
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) %
        1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&time, &local);

    std::ostringstream line;
    line << std::put_time(&local, "%Y-%m-%dT%H:%M:%S") << '.'
         << std::setfill('0') << std::setw(3) << milliseconds.count()
         << ' ' << log_level_name(level) << ' ' << message;
    const std::lock_guard<std::mutex> lock(g_log_mutex);
    std::clog << line.str() << '\n';
}

}  // namespace double_ok_gesture
