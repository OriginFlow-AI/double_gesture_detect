#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <thread>

#include "double_ok_gesture/hevc_async_reader.hpp"

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

int main(int argc, char** argv) {
    std::string device = "/dev/video0";
    int width = 2560;
    int height = 1024;
    double fps = 30.0;

    if (argc > 1) device = argv[1];
    if (argc > 2) width = std::stoi(argv[2]);
    if (argc > 3) height = std::stoi(argv[3]);
    if (argc > 4) fps = std::stod(argv[4]);

    std::cout << "=== HEVC Frame Capture Test ===" << std::endl;
    std::cout << "Device: " << device << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        auto reader = std::make_unique<double_ok_gesture::HevcAsyncReader>(
            device, width, height, fps);

        std::cout << "Backend: " << reader->info().backend << std::endl;
        std::cout << "FourCC: " << reader->info().fourcc << std::endl;
        std::cout << "Starting capture (Ctrl+C to stop)..." << std::endl;

        auto start_time = std::chrono::steady_clock::now();
        int frame_count = 0;
        int error_count = 0;

        while (g_running) {
            auto frame = reader->read();
            if (frame) {
                frame_count++;
                if (frame_count % 30 == 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time).count();
                    double actual_fps = frame_count * 1000.0 / elapsed;
                    std::cout << "Frames: " << frame_count
                              << " | FPS: " << actual_fps
                              << " | Size: " << frame->cols << "x" << frame->rows
                              << " | Type: " << frame->type() << std::endl;
                }
            } else {
                error_count++;
                if (error_count <= 5) {
                    std::cout << "No frame available (count: " << error_count << ")" << std::endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        reader->stop();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        double actual_fps = frame_count * 1000.0 / elapsed;

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Total frames: " << frame_count << std::endl;
        std::cout << "Average FPS: " << actual_fps << std::endl;
        std::cout << "Dropped frames: " << error_count << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
