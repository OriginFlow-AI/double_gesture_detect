#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <sys/mman.h>

// 前向声明 - 避免直接引用 MPP 头文件
struct MppCtx;
struct MppApi;
struct MppBuffer;
struct MppFrame;
struct _MppPacket;
typedef _MppPacket MppPacket;
typedef unsigned int MppFrameFormat;
typedef int MPP_RET;

namespace double_ok_gesture {

/**
 * HevcMppReader - 使用 MPP + V4L2 实现 H.265 硬件解码
 *
 * 该类通过 V4L2 从摄像头读取 H.265 压缩数据，然后使用瑞芯微 MPP
 * 硬件解码器进行解码，输出 BGR 格式的图像帧。
 *
 * 使用方法：
 *   auto reader = std::make_unique<HevcMppReader>(
 *       "/dev/video0", 1920, 1080, 30.0);
 *   while (auto frame = reader->read()) {
 *       // 处理 frame
 *   }
 */
class HevcMppReader {
public:
    /**
     * @param device V4L2 设备路径，如 "/dev/video0"
     * @param width 视频宽度
     * @param height 视频高度
     * @param fps 帧率
     * @param max_queue_size 解码后帧队列最大长度
     */
    HevcMppReader(
        const std::string& device,
        int width,
        int height,
        double fps,
        size_t max_queue_size = 2);

    ~HevcMppReader();

    HevcMppReader(const HevcMppReader&) = delete;
    HevcMppReader& operator=(const HevcMppReader&) = delete;
    HevcMppReader(HevcMppReader&&) = delete;
    HevcMppReader& operator=(HevcMppReader&&) = delete;

    /**
     * 读取解码后的下一帧
     * @return 解码后的 BGR 图像，如果队列为空则返回 std::nullopt
     */
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

private:
    void decoder_loop();

    MppCtx* ctx_ = nullptr;
    MppApi* mpi_ = nullptr;
    void* buffer_group_ = nullptr;
    std::thread decoder_thread_;
    std::queue<cv::Mat> frame_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_queue_size_ = 2;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopped_{false};
    Info info_;

    int v4l2_fd_ = -1;
    struct V4L2Buffer {
        void* start = MAP_FAILED;
        size_t length = 0;
    };
    std::vector<V4L2Buffer> v4l2_buffers_;
};

}  // namespace double_ok_gesture
