#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

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

void maybe_open_browser(const std::filesystem::path& path) {
    const pid_t launcher = ::fork();
    if (launcher < 0) {
        throw std::runtime_error(
            "Cannot launch xdg-open: " + std::string(std::strerror(errno)));
    }
    if (launcher == 0) {
        const pid_t detached = ::fork();
        if (detached < 0) {
            ::_exit(127);
        }
        if (detached > 0) {
            ::_exit(0);
        }
        (void)::setsid();
        const int null_fd = ::open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            (void)::dup2(null_fd, STDOUT_FILENO);
            (void)::dup2(null_fd, STDERR_FILENO);
            ::close(null_fd);
        }
        const std::string value = path.string();
        ::execlp(
            "xdg-open",
            "xdg-open",
            value.c_str(),
            static_cast<char*>(nullptr));
        ::_exit(127);
    }

    int status = 0;
    while (::waitpid(launcher, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::runtime_error(
                "Cannot wait for xdg-open launcher: " +
                std::string(std::strerror(errno)));
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("Cannot detach xdg-open launcher");
    }
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
