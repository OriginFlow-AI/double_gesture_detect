#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/report.hpp"

namespace {

struct Args {
    double_ok_gesture::GuiReportRequest report;
    bool open = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + key);
            }
            return argv[++i];
        };
        if (key == "--config") {
            args.report.config = next();
        } else if (key == "--csv") {
            args.report.csv = next();
        } else if (key == "--model") {
            args.report.model = next();
        } else if (key == "--output") {
            args.report.output = next();
        } else if (key == "--open") {
            args.open = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    return args;
}

std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out += "'";
    return out;
}

void maybe_open_browser(const std::filesystem::path& path) {
    const std::string command = "xdg-open " + shell_quote(path.string()) + " >/dev/null 2>&1 &";
    const int exit_code = std::system(command.c_str());
    (void)exit_code;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        double_ok_gesture::write_gui_report(args.report);
        std::cout << args.report.output.string() << '\n';
        if (args.open) {
            maybe_open_browser(std::filesystem::absolute(args.report.output));
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
