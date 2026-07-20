#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include <opencv2/core.hpp>

struct _GstAppSink;
struct _GstPipeline;
using GstPipeline = _GstPipeline;
using GstAppSink = _GstAppSink;

namespace double_ok_gesture {

class HevcAsyncReader {
public:
    HevcAsyncReader(
        const std::string& device,
        int width,
        int height,
        double fps,
        size_t max_queue_size = 2);

    ~HevcAsyncReader();

    HevcAsyncReader(const HevcAsyncReader&) = delete;
    HevcAsyncReader& operator=(const HevcAsyncReader&) = delete;
    HevcAsyncReader(HevcAsyncReader&&) = delete;
    HevcAsyncReader& operator=(HevcAsyncReader&&) = delete;

    std::optional<cv::Mat> read();

    struct Info {
        std::string source;
        std::string backend;
        int width = 0;
        int height = 0;
        double fps = 0.0;
        std::string fourcc;
    };
    const Info& info() const { return info_; }

    bool is_running() const { return running_.load(std::memory_order_relaxed); }
    void stop();

    void on_new_sample(GstAppSink* appsink);

private:
    void decoder_loop();

    GstPipeline* pipeline_ = nullptr;
    GstAppSink* appsink_ = nullptr;
    std::thread decoder_thread_;
    std::queue<cv::Mat> frame_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_queue_size_ = 2;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopped_{false};
    Info info_;
};

}  // namespace double_ok_gesture
