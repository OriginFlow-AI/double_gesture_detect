#include "double_ok_gesture/hevc_async_reader.hpp"

#include <stdexcept>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "double_ok_gesture/runtime.hpp"

extern "C" {
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/app.h>
}

namespace double_ok_gesture {
namespace {

cv::Mat gst_sample_to_mat(GstSample* sample) {
    if (!sample) {
        return cv::Mat{};
    }

    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps) {
        gst_sample_unref(sample);
        return cv::Mat{};
    }

    GstStructure* s = gst_caps_get_structure(caps, 0);
    if (!s) {
        gst_sample_unref(sample);
        return cv::Mat{};
    }

    gint width, height, stride = 0;
    if (!gst_structure_get_int(s, "width", &width) ||
        !gst_structure_get_int(s, "height", &height)) {
        gst_sample_unref(sample);
        return cv::Mat{};
    }
    gst_structure_get_int(s, "stride", &stride);
    if (stride == 0) {
        gst_structure_get_int(s, "bytesperline", &stride);
    }
    if (stride == 0) {
        stride = width;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return cv::Mat{};
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return cv::Mat{};
    }

    static bool logged = false;
    if (!logged) {
        const gchar* fmt = gst_structure_get_string(s, "format");
        log_message(LogLevel::Info, "Frame: " + std::to_string(width) + "x" + std::to_string(height) +
            " stride=" + std::to_string(stride) + " format=" + (fmt ? fmt : "null") +
            " size=" + std::to_string(map.size));
        logged = true;
    }

    cv::Mat mat;
    const gchar* format = gst_structure_get_string(s, "format");

    // RGB format from videoconvert - already in BGR order after cvtColor, just copy
    if (format && strcmp(format, "RGB") == 0) {
        if (stride == width * 3) {
            cv::Mat rgb(height, width, CV_8UC3, map.data);
            cv::cvtColor(rgb, mat, cv::COLOR_RGB2BGR);
        } else {
            cv::Mat rgb_with_stride(height, stride, CV_8UC3, map.data);
            cv::Mat rgb_cropped = rgb_with_stride(cv::Rect(0, 0, width, height));
            cv::cvtColor(rgb_cropped, mat, cv::COLOR_RGB2BGR);
        }
    } else if (format && strcmp(format, "I420") == 0) {
        cv::Mat yuv(height * 3 / 2, width, CV_8UC1, map.data);
        cv::cvtColor(yuv, mat, cv::COLOR_YUV2BGR_I420);
    } else if (format && (strcmp(format, "RGBA") == 0 || strcmp(format, "BGRA") == 0)) {
        cv::Mat rgba(height, width, CV_8UC4, map.data);
        if (strcmp(format, "BGRA") == 0) {
            cv::cvtColor(rgba, mat, cv::COLOR_BGRA2BGR);
        } else {
            cv::cvtColor(rgba, mat, cv::COLOR_RGBA2BGR);
        }
    } else if (format && strcmp(format, "RGBx") == 0) {
        cv::Mat rgbx(height, width, CV_8UC4, map.data);
        cv::cvtColor(rgbx, mat, cv::COLOR_RGBA2BGR);
    } else {
        // Default fallback - try NV12
        cv::Mat yuv(height * 3 / 2, width, CV_8UC1, map.data);
        cv::cvtColor(yuv, mat, cv::COLOR_YUV2BGR_NV12);
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    if (mat.empty()) {
        return cv::Mat{};
    }
    return mat.clone();
}

}  // namespace

HevcAsyncReader::HevcAsyncReader(
    const std::string& device,
    const int width,
    const int height,
    const double fps,
    const size_t max_queue_size)
    : max_queue_size_(max_queue_size) {

    gst_init(nullptr, nullptr);

    log_message(LogLevel::Info, "Opening HEVC camera via GStreamer: " + device);

    int right_crop = width / 2;
    std::string pipeline_str = "v4l2src device=" + device +
        " ! video/x-h265,stream-format=byte-stream,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1"
        " ! h265parse"
        " ! queue leaky=2 max-size-buffers=1"
        " ! mppvideodec"
        " ! videoconvert"
        " ! video/x-raw,format=RGB"
        " ! appsink name=sink emit-signals=false sync=false";

    log_message(LogLevel::Info, "GStreamer pipeline: " + pipeline_str);

    GError* error = nullptr;
    pipeline_ = GST_PIPELINE(gst_parse_launch(pipeline_str.c_str(), &error));
    if (error) {
        log_message(LogLevel::Error, "Failed to create GStreamer pipeline: " + std::string(error->message));
        g_error_free(error);
        pipeline_ = nullptr;
        return;
    }

    GstElement* appsink_elem = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_elem) {
        log_message(LogLevel::Error, "Failed to get appsink from pipeline");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return;
    }
    appsink_ = GST_APP_SINK(appsink_elem);
    gst_object_unref(appsink_elem);

    GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(pipeline_), GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        log_message(LogLevel::Error, "Failed to start GStreamer pipeline");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return;
    }

    if (appsink_) {
        log_message(LogLevel::Info, "appsink_ initialized successfully");
    } else {
        log_message(LogLevel::Error, "appsink_ is NULL after initialization");
    }

    info_.source = device;
    info_.backend = "GStreamer/mppvideodec+ videoconvert";
    info_.width = width;  // Original width, crop done in Qt
    info_.height = height;
    info_.fps = fps;
    info_.fourcc = "HEVC";

    log_message(
        LogLevel::Info,
        "HEVC camera ready: source=" + info_.source + ", backend=" + info_.backend +
            ", size=" + std::to_string(info_.width) + "x" + std::to_string(info_.height) +
            ", fps=" + std::to_string(info_.fps) + ", fourcc=" + info_.fourcc);

    running_.store(true, std::memory_order_relaxed);
    stopped_.store(false, std::memory_order_relaxed);
    decoder_thread_ = std::thread(&HevcAsyncReader::decoder_loop, this);
}

HevcAsyncReader::~HevcAsyncReader() {
    stop();
}

void HevcAsyncReader::stop() {
    if (stopped_.load(std::memory_order_acquire)) {
        return;
    }
    running_.store(false, std::memory_order_relaxed);
    if (decoder_thread_.joinable()) {
        decoder_thread_.join();
    }
    if (pipeline_) {
        gst_element_set_state(GST_ELEMENT(pipeline_), GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    stopped_.store(true, std::memory_order_release);
}

void HevcAsyncReader::decoder_loop() {
    log_message(LogLevel::Info, "HEVC decoder thread started");

    int frame_count = 0;
    int empty_count = 0;
    int loop_iterations = 0;
    auto last_report = std::chrono::steady_clock::now();
    int64_t pull_time_total = 0, convert_time_total = 0, queue_time_total = 0;

    while (running_.load(std::memory_order_acquire)) {
        loop_iterations++;

        if (!appsink_) {
            if (frame_count < 3) log_message(LogLevel::Debug, std::string("appsink_ is null"));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto t1 = std::chrono::steady_clock::now();
        GstSample* sample = nullptr;
        try {
            sample = gst_app_sink_pull_sample(appsink_);
        } catch (const std::exception& e) {
            log_message(LogLevel::Error, std::string("GStreamer pull error: ") + e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto t2 = std::chrono::steady_clock::now();

        if (!sample) {
            empty_count++;
            continue;
        }

        empty_count = 0;
        pull_time_total += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        cv::Mat frame;
        try {
            frame = gst_sample_to_mat(sample);
        } catch (const std::exception& e) {
            log_message(LogLevel::Error, std::string("Frame conversion error: ") + e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto t3 = std::chrono::steady_clock::now();
        convert_time_total += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        if (!frame.empty()) {
            auto t4 = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(mutex_);
            if (frame_queue_.size() >= max_queue_size_) {
                frame_queue_.pop();
            }
            frame_queue_.push(frame.clone());
            ++frame_count;
            queue_time_total += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t4).count();
        }

        // Report every 2 seconds
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count();
        if (elapsed >= 2 && frame_count > 0) {
            log_message(LogLevel::Info, "Decode: pull_avg=" +
                std::to_string(pull_time_total / frame_count / 1000) + "ms, convert_avg=" +
                std::to_string(convert_time_total / frame_count / 1000) + "ms, queue_avg=" +
                std::to_string(queue_time_total / frame_count / 1000) + "ms, total=" +
                std::to_string(frame_count) + " frames");
            last_report = now;
            pull_time_total = convert_time_total = queue_time_total = 0;
        }
    }

    log_message(LogLevel::Info, "HEVC decoder thread stopped after " + std::to_string(frame_count) + " frames");
}

std::optional<cv::Mat> HevcAsyncReader::read() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_queue_.empty()) {
        return std::nullopt;
    }
    cv::Mat frame = frame_queue_.front();
    frame_queue_.pop();
    return frame;
}

}  // namespace double_ok_gesture
