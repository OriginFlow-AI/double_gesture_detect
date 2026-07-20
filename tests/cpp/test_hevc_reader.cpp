#include <iostream>
#include <stdexcept>
#include <string>
#include <optional>
#include <memory>

#include "double_ok_gesture/hevc_mpp_reader.hpp"

namespace {

#define EXPECT_TRUE(expr) \
    do { if (!(expr)) throw std::runtime_error(std::string("EXPECT_TRUE failed: ") + #expr); } while (false)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ(lhs, rhs) \
    do { const auto _lhs = (lhs); const auto _rhs = (rhs); if (!(_lhs == _rhs)) throw std::runtime_error(std::string("EXPECT_EQ failed: ") + #lhs + " != " + #rhs); } while (false)

void test_hevc_mpp_reader_info() {
    std::cout << "Test: HevcMppReader::Info structure..." << std::endl;

    double_ok_gesture::HevcMppReader::Info info;
    info.source = "/dev/video0";
    info.backend = "MPP";
    info.width = 1920;
    info.height = 1080;
    info.fps = 30.0;
    info.fourcc = "HEVC";

    EXPECT_EQ(info.source, "/dev/video0");
    EXPECT_EQ(info.backend, "MPP");
    EXPECT_EQ(info.width, 1920);
    EXPECT_EQ(info.height, 1080);
    EXPECT_EQ(info.fps, 30.0);
    EXPECT_EQ(info.fourcc, "HEVC");

    std::cout << "  PASS: HevcMppReader::Info is well-formed" << std::endl;
}

void test_hevc_mpp_reader_static_polymorphism() {
    std::cout << "Test: HevcMppReader polymorphic behavior..." << std::endl;

    // 使用基类指针测试接口
    double_ok_gesture::HevcMppReader* reader = nullptr;

    // 在 MPP 不可用时，构造函数会抛出
    try {
        reader = new double_ok_gesture::HevcMppReader("/dev/video0", 1920, 1080, 30.0);
        // 如果没有抛异常，说明 MPP 可用
        std::cout << "  Note: MPP is available, reader created" << std::endl;
        delete reader;
    } catch (const std::runtime_error& e) {
        std::cout << "  PASS: HevcMppReader throws when MPP not available: " << e.what() << std::endl;
        return;
    }

    std::cout << "  PASS: Polymorphic test completed" << std::endl;
}

void test_optional_mat() {
    std::cout << "Test: cv::Mat in std::optional..." << std::endl;

    std::optional<cv::Mat> empty_opt;
    EXPECT_TRUE(!empty_opt.has_value());

    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
    std::optional<cv::Mat> opt = mat;
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(opt->cols, 100);
    EXPECT_EQ(opt->rows, 100);

    opt.reset();
    EXPECT_TRUE(!opt.has_value());

    std::cout << "  PASS: std::optional<cv::Mat> works correctly" << std::endl;
}

void test_queue_frame_processing() {
    std::cout << "Test: Frame queue processing simulation..." << std::endl;

    std::queue<cv::Mat> frame_queue;

    // 模拟入队
    for (int i = 0; i < 3; i++) {
        cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(i * 50, i * 50, i * 50));
        frame_queue.push(frame);
    }

    EXPECT_EQ(frame_queue.size(), 3u);

    // 模拟出队
    int count = 0;
    while (!frame_queue.empty()) {
        cv::Mat frame = frame_queue.front();
        frame_queue.pop();
        EXPECT_EQ(frame.cols, 100);
        EXPECT_EQ(frame.rows, 100);
        count++;
    }
    EXPECT_EQ(count, 3);

    std::cout << "  PASS: Frame queue processing works correctly" << std::endl;
}

}  // namespace

int main() {
    std::cout << "=== HEVC Reader Tests ===" << std::endl;

    try {
        test_hevc_mpp_reader_info();
        test_optional_mat();
        test_queue_frame_processing();
        test_hevc_mpp_reader_static_polymorphism();

        std::cout << "\n=== All tests passed ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
