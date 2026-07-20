#include "double_ok_gesture/hevc_mpp_reader.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/select.h>

#include <cstring>
#include <stdexcept>
#include <cerrno>

#include <opencv2/imgproc.hpp>

#include "double_ok_gesture/runtime.hpp"

#if defined(DOUBLE_OK_HAS_MPP) && DOUBLE_OK_HAS_MPP

extern "C" {
#include <rk_mpi.h>
#include <rk_type.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <mpp_buffer.h>
}

namespace double_ok_gesture {
namespace {

constexpr int DEFAULT_POOL_DEPTH = 8;
constexpr int MAX_EMPTY_POLLS = 100;
constexpr int V4L2_TIMEOUT_MS = 10;

}  // namespace

HevcMppReader::HevcMppReader(
    const std::string& device,
    const int width,
    const int height,
    const double fps,
    const size_t max_queue_size)
    : max_queue_size_(max_queue_size) {

    log_message(LogLevel::Info, "Opening HEVC camera via MPP: " + device);

    info_.source = device;
    info_.backend = "MPP";
    info_.width = width;
    info_.height = height;
    info_.fps = fps;
    info_.fourcc = "HEVC";

    // 创建 MPP 解码器上下文
    MPP_RET ret = mpp_create(&ctx_, &mpi_);
    if (ret != MPP_OK) {
        log_message(LogLevel::Error, "mpp_create failed: ret=" + std::to_string(ret));
        throw std::runtime_error("mpp_create failed");
    }

    // 初始化 MPP 解码器为 H.265 硬解
    ret = mpp_init(ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingHEVC);
    if (ret != MPP_OK) {
        log_message(LogLevel::Error, "mpp_init failed: ret=" + std::to_string(ret));
        mpp_destroy(ctx_);
        throw std::runtime_error("mpp_init failed");
    }

    // 打开 V4L2 设备
    v4l2_fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (v4l2_fd_ < 0) {
        log_message(LogLevel::Error, "Failed to open V4L2 device: " + device + " - " + strerror(errno));
        mpp_destroy(ctx_);
        throw std::runtime_error("Failed to open V4L2 device: " + device);
    }

    // 配置 V4L2 为 H.265 格式
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = width;
    fmt.fmt.pix_mp.height = height;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H265;
    fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;
    if (ioctl(v4l2_fd_, VIDIOC_S_FMT, &fmt) < 0) {
        log_message(LogLevel::Error, "VIDIOC_S_FMT failed: " + std::string(strerror(errno)));
        close(v4l2_fd_);
        mpp_destroy(ctx_);
        throw std::runtime_error("VIDIOC_S_FMT failed");
    }

    // 请求 V4L2 缓冲区
    struct v4l2_requestbuffers reqbufs = {};
    reqbufs.count = DEFAULT_POOL_DEPTH;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbufs.memory = V4L2_MEMORY_MMAP;
    if (ioctl(v4l2_fd_, VIDIOC_REQBUFS, &reqbufs) < 0) {
        log_message(LogLevel::Error, "VIDIOC_REQBUFS failed: " + std::string(strerror(errno)));
        close(v4l2_fd_);
        mpp_destroy(ctx_);
        throw std::runtime_error("VIDIOC_REQBUFS failed");
    }

    log_message(LogLevel::Info, "V4L2 buffers requested: " + std::to_string(reqbufs.count));

    // 映射和入队缓冲区
    v4l2_buffers_.resize(reqbufs.count);
    for (unsigned int i = 0; i < reqbufs.count; ++i) {
        struct v4l2_buffer buf = {};
        struct v4l2_plane planes[2] = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 2;
        if (ioctl(v4l2_fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            log_message(LogLevel::Error, "VIDIOC_QUERYBUF failed for buf " + std::to_string(i));
            close(v4l2_fd_);
            mpp_destroy(ctx_);
            throw std::runtime_error("VIDIOC_QUERYBUF failed");
        }
        v4l2_buffers_[i].length = buf.m.planes->length;
        v4l2_buffers_[i].start = mmap(NULL, buf.m.planes->length,
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       v4l2_fd_, buf.m.planes->m.mem_offset);
        if (v4l2_buffers_[i].start == MAP_FAILED) {
            log_message(LogLevel::Error, "mmap failed for buf " + std::to_string(i));
            close(v4l2_fd_);
            mpp_destroy(ctx_);
            throw std::runtime_error("mmap failed");
        }
        // 入队
        if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buf) < 0) {
            log_message(LogLevel::Error, "VIDIOC_QBUF failed for buf " + std::to_string(i));
            close(v4l2_fd_);
            mpp_destroy(ctx_);
            throw std::runtime_error("VIDIOC_QBUF failed");
        }
    }

    // 开始采集
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(v4l2_fd_, VIDIOC_STREAMON, &type) < 0) {
        log_message(LogLevel::Error, "VIDIOC_STREAMON failed: " + std::string(strerror(errno)));
        close(v4l2_fd_);
        mpp_destroy(ctx_);
        throw std::runtime_error("VIDIOC_STREAMON failed");
    }

    log_message(LogLevel::Info, "MPP decoder initialized successfully: " + device +
        " " + std::to_string(width) + "x" + std::to_string(height));

    running_.store(true, std::memory_order_relaxed);
    stopped_.store(false, std::memory_order_relaxed);
    decoder_thread_ = std::thread(&HevcMppReader::decoder_loop, this);
}

HevcMppReader::~HevcMppReader() {
    stop();
}

void HevcMppReader::stop() {
    if (stopped_.load(std::memory_order_acquire)) {
        return;
    }
    running_.store(false, std::memory_order_relaxed);
    if (decoder_thread_.joinable()) {
        decoder_thread_.join();
    }

    // 停止 V4L2 采集
    if (v4l2_fd_ >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        ioctl(v4l2_fd_, VIDIOC_STREAMOFF, &type);
        for (auto& buf : v4l2_buffers_) {
            if (buf.start != MAP_FAILED) {
                munmap(buf.start, buf.length);
            }
        }
        v4l2_buffers_.clear();
        close(v4l2_fd_);
        v4l2_fd_ = -1;
    }

    // 清理 MPP
    if (ctx_) {
        mpi_->reset(ctx_);
        mpp_destroy(ctx_);
        ctx_ = nullptr;
        mpi_ = nullptr;
    }

    stopped_.store(true, std::memory_order_release);
    log_message(LogLevel::Info, "MPP decoder stopped");
}

void HevcMppReader::decoder_loop() {
    log_message(LogLevel::Info, "MPP decoder thread started");

    int frame_count = 0;
    int empty_count = 0;

    while (running_.load(std::memory_order_acquire)) {
        // 从 V4L2 读取一帧压缩数据
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(v4l2_fd_, &fds);
        struct timeval tv = {0, V4L2_TIMEOUT_MS * 1000};  // 10ms timeout
        int r = select(v4l2_fd_ + 1, &fds, NULL, NULL, &tv);
        if (r <= 0) {
            continue;
        }

        // Dequeue V4L2 buffer
        struct v4l2_buffer buf = {};
        struct v4l2_plane planes[2] = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.m.planes = planes;
        buf.length = 2;
        if (ioctl(v4l2_fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) {
                continue;
            }
            log_message(LogLevel::Error, "VIDIOC_DQBUF failed: " + std::string(strerror(errno)));
            break;
        }

        // 获取压缩数据
        void* compressed_data = v4l2_buffers_[buf.index].start;
        size_t compressed_size = buf.m.planes->bytesused;

        if (compressed_size == 0) {
            // 空帧，重新入队
            ioctl(v4l2_fd_, VIDIOC_QBUF, &buf);
            continue;
        }

        // 创建 MPP packet 发送压缩数据
        MppPacket packet = nullptr;
        MPP_RET ret = mpp_packet_init(&packet, compressed_data, compressed_size);
        if (ret != MPP_OK) {
            log_message(LogLevel::Error, "mpp_packet_init failed: ret=" + std::to_string(ret));
            ioctl(v4l2_fd_, VIDIOC_QBUF, &buf);
            continue;
        }

        // 设置 packet 参数
        mpp_packet_set_pts(packet, buf.timestamp.tv_sec * 1000000LL + buf.timestamp.tv_usec);

        // 解码 - 使用同步接口
        MppFrame frame = nullptr;
        ret = mpi_->decode(ctx_, packet, &frame);
        mpp_packet_deinit(&packet);

        if (ret != MPP_OK) {
            log_message(LogLevel::Error, "MPP decode failed: ret=" + std::to_string(ret));
            ioctl(v4l2_fd_, VIDIOC_QBUF, &buf);
            continue;
        }

        // Re-queue buffer 立即重新入队
        ioctl(v4l2_fd_, VIDIOC_QBUF, &buf);

        if (!frame) {
            empty_count++;
            if (empty_count > MAX_EMPTY_POLLS) {
                log_message(LogLevel::Debug, "MPP: too many empty frames, continuing...");
                empty_count = 0;
            }
            continue;
        }

        empty_count = 0;

        // 获取解码后的 YUV 数据
        MppBuffer buffer = mpp_frame_get_buffer(frame);
        if (!buffer) {
            mpp_frame_deinit(&frame);
            continue;
        }

        // 获取帧信息
        int frame_width = mpp_frame_get_width(frame);
        int frame_height = mpp_frame_get_height(frame);
        int hor_stride = mpp_frame_get_hor_stride(frame);
        int ver_stride = mpp_frame_get_ver_stride(frame);
        MppFrameFormat fmt = mpp_frame_get_fmt(frame);

        // 转换 YUV 到 BGR
        void* yuv_data = mpp_buffer_get_ptr(buffer);
        cv::Mat yuv_mat;
        if (fmt == MPP_FMT_YUV420P) {
            // YYYYYYYY UU VV (I420)
            cv::Mat yuv_planar(frame_height * 3 / 2, hor_stride, CV_8UC1, yuv_data);
            cv::cvtColor(yuv_planar(cv::Rect(0, 0, frame_width, frame_height * 3 / 2)),
                        yuv_mat, cv::COLOR_YUV2BGR_I420);
        } else if (fmt == MPP_FMT_YUV420SP) {
            // NV12: YYYYYYYY UVUVUVUV
            cv::Mat yuv_sp(frame_height * 3 / 2, hor_stride, CV_8UC1, yuv_data);
            cv::cvtColor(yuv_sp(cv::Rect(0, 0, frame_width, frame_height * 3 / 2)),
                        yuv_mat, cv::COLOR_YUV2BGR_NV12);
        } else if (fmt == MPP_FMT_YUV420SP_VU) {
            // NV21: YYYYYYYY VUVUVUVU
            cv::Mat yuv_sp(frame_height * 3 / 2, hor_stride, CV_8UC1, yuv_data);
            cv::cvtColor(yuv_sp(cv::Rect(0, 0, frame_width, frame_height * 3 / 2)),
                        yuv_mat, cv::COLOR_YUV2BGR_NV21);
        } else {
            log_message(LogLevel::Debug, "MPP: unsupported format: " + std::to_string(fmt));
            mpp_frame_deinit(&frame);
            continue;
        }

        mpp_frame_deinit(&frame);

        if (!yuv_mat.empty()) {
            // 调整大小到目标分辨率
            cv::Mat resized;
            if (yuv_mat.cols != info_.width || yuv_mat.rows != info_.height) {
                cv::resize(yuv_mat, resized, cv::Size(info_.width, info_.height));
            } else {
                resized = yuv_mat;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            if (frame_queue_.size() >= max_queue_size_) {
                frame_queue_.pop();
            }
            frame_queue_.push(resized);
            ++frame_count;
            if (frame_count <= 5) {
                log_message(LogLevel::Debug, "MPP Frame decoded: " + std::to_string(frame_count));
            }
        }
    }

    log_message(LogLevel::Info, "MPP decoder thread stopped after " + std::to_string(frame_count) + " frames");
}

std::optional<cv::Mat> HevcMppReader::read() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_queue_.empty()) {
        return std::nullopt;
    }
    cv::Mat frame = frame_queue_.front();
    frame_queue_.pop();
    return frame;
}

}  // namespace double_ok_gesture

#else  // DOUBLE_OK_HAS_MPP

// MPP 未启用时的空实现
namespace double_ok_gesture {

HevcMppReader::HevcMppReader(
    const std::string& device,
    const int width,
    const int height,
    const double fps,
    const size_t max_queue_size)
    : max_queue_size_(max_queue_size) {
    (void)device;
    (void)width;
    (void)height;
    (void)fps;
    log_message(LogLevel::Error, "HevcMppReader: MPP not available in this build");
    throw std::runtime_error("HevcMppReader requires MPP (RK3576)");
}

HevcMppReader::~HevcMppReader() = default;

void HevcMppReader::stop() {
    stopped_.store(true, std::memory_order_release);
}

std::optional<cv::Mat> HevcMppReader::read() {
    return std::nullopt;
}

}  // namespace double_ok_gesture

#endif  // DOUBLE_OK_HAS_MPP
