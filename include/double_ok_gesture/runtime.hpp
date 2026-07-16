#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <string_view>

namespace double_ok_gesture {

struct RuntimeSnapshot {
    double fps = 0.0;
    double processing_ms = 0.0;
    std::size_t frame_count = 0;
    double inference_ms = 0.0;
};

class RuntimeMetrics {
public:
    explicit RuntimeMetrics(std::size_t window_size = 60);
    RuntimeSnapshot update(double frame_started, double frame_finished);
    RuntimeSnapshot update(double frame_started);

private:
    std::deque<double> timestamps_;
    std::size_t window_size_ = 60;
    std::size_t frame_count_ = 0;
};

double monotonic_seconds();

enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error,
};

void configure_logging(const std::string& level);
void log_message(LogLevel level, std::string_view message);

}  // namespace double_ok_gesture
