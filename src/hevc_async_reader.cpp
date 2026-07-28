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

    // 架构 v2：解码与色彩转换分离
    //   - decoder_loop 线程中只做轻量映射（不执行 cvtColor），保证每帧解码
    //   - consumer 线程按需执行 cvtColor (NV12/I420 → BGR) + 推理
    // RGB 路径：仍直接转 BGR（consumer 期望 3 通道 BGR）
    if (format && strcmp(format, "RGB") == 0) {
        if (stride == width * 3) {
            cv::Mat rgb(height, width, CV_8UC3, map.data);
            cv::cvtColor(rgb, mat, cv::COLOR_RGB2BGR);
        } else {
            cv::Mat rgb_with_stride(height, stride, CV_8UC3, map.data);
            cv::Mat rgb_cropped = rgb_with_stride(cv::Rect(0, 0, width, height));
            cv::cvtColor(rgb_cropped, mat, cv::COLOR_RGB2BGR);
        }
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
        // NV12 / I420 路径：直接返回 NV12 (YUV420) 数据，consumer 按需 cvtColor
        // 标记为 CV_8UC1 + h*3/2 行：这是 OpenCV 表示 YUV420 紧凑平面的标准做法
        // 外部代码通过 channels()==1 && rows == height*3/2 判断是 YUV 格式
        mat = cv::Mat(height * 3 / 2, width, CV_8UC1, map.data);
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
    const size_t max_queue_size,
    const SourceType source_type,
    const bool loop_file)
    : max_queue_size_(max_queue_size),
      source_type_(source_type),
      loop_file_(loop_file) {

    gst_init(nullptr, nullptr);

    log_message(
        LogLevel::Info,
        std::string(source_type == SourceType::FILE ? "Opening HEVC file: " : "Opening HEVC camera via GStreamer: ") + device +
        (loop_file ? " (loop)" : ""));

    int right_crop = width / 2;
    (void)right_crop;  // 抑制未使用变量警告：实际裁剪通过 GStreamer videocrop 完成
    // 摄像头拼接模式时取左半（在解码后立即裁剪，避免重复占用解码资源）
    //   2560x1024 -> 1280x1024（左半）
    //   3840x1080 -> 1920x1080（左半）
    std::string crop_filter;
    if (width == 2560 && height == 1024) {
        crop_filter = " ! videocrop right=1280 top=0 bottom=0";
    } else if (width == 3840 && height == 1080) {
        crop_filter = " ! videocrop right=1920 top=0 bottom=0";
    }
    // 源输入：v4l2src (实时摄像头) 或 filesrc (h265 文件回放)
    std::string source_str;
    if (source_type == SourceType::FILE) {
        // loop_file=true 时 filesrc loop=true 自动循环播放
        source_str = "filesrc location=" + device + (loop_file ? " loop=true" : "");
    } else {
        source_str = "v4l2src device=" + device;
    }
    std::string pipeline_str = source_str +
        " ! video/x-h265,stream-format=byte-stream,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1"
        " ! h265parse"
        // 架构说明（v2）：解码与色彩转换分离
        //   - pipeline 端：仅做解码 + crop（保持 NV12，无 videoconvert），保证 mppvideodec 不被反压
        //   - consumer 端：每 N 帧（detection_skip_frames）做一次 NV12→BGR + 手势推理
        //   - 收益：mppvideodec 可持续以 30 fps 输出，videoconvert 慢路径只在 1/N 帧执行
        // queue 配置：保留 4 帧缓冲 + leaky=2 (下游丢旧)，避免反压时丢 I 帧
        " ! queue leaky=2 max-size-buffers=4 max-size-time=0 max-size-bytes=0"
        " ! mppvideodec"
        // crop 减少后续 consumer 的处理量（在 mppvideodec 之后、videoconvert 之前）
        + crop_filter +
        // 不做 videoconvert：appsink 接收 NV12，consumer 按需 NV12→BGR
        " ! video/x-raw,format=NV12"
        // appsink 配置：max-buffers=1 + drop=true，消费者慢时主动丢旧 buffer
        " ! appsink name=sink max-buffers=1 drop=true emit-signals=false sync=false";

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
            // 文件源 EOS：仅当 loop_file=true 时 seek 重置循环（仅 file 模式）
            // 注：mppvideodec seek 后内部状态可能损坏（持续 0 帧），生产环境建议
            //     用 --max-frames 配合外部脚本重启，更可靠。
            if (loop_file_ && source_type_ == SourceType::FILE && pipeline_ != nullptr) {
                log_message(LogLevel::Info, "File EOS reached, seeking back to 0 for loop playback");
                gst_element_seek(
                    reinterpret_cast<GstElement*>(pipeline_),
                    1.0,           // rate
                    GST_FORMAT_TIME,
                    GST_SEEK_FLAG_FLUSH,
                    GST_SEEK_TYPE_SET, 0,    // start
                    GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);  // end
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
