#include <iostream>
#include <stdexcept>

#include "double_ok_gesture/demo_app.hpp"

int main(int argc, char** argv) {
    try {
        const auto args = double_ok_gesture::parse_demo_args(argc, argv);
        if (args.list_cameras) {
            return double_ok_gesture::write_demo_camera_list(std::cout);
        }
        return double_ok_gesture::run_demo_headless(args, std::cout, std::cerr);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
