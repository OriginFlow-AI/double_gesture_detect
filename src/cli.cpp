#include "double_ok_gesture/cli.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <system_error>

namespace double_ok_gesture {
namespace {

[[noreturn]] void invalid_argument_value(
    std::string_view value,
    std::string_view option,
    const char* expected) {
    throw std::invalid_argument(
        std::string(option) + " requires " + expected + "; got '" +
        std::string(value) + "'");
}

template <typename T>
T parse_integral(
    std::string_view value,
    std::string_view option,
    const char* expected) {
    T result{};
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size()) {
        invalid_argument_value(value, option, expected);
    }
    return result;
}

}  // namespace

int parse_int_argument(std::string_view value, std::string_view option) {
    return parse_integral<int>(value, option, "an integer");
}

double parse_finite_double_argument(
    std::string_view value,
    std::string_view option) {
    double result = 0.0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result,
        std::chars_format::general);
    if (value.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size() || !std::isfinite(result)) {
        invalid_argument_value(value, option, "a finite number");
    }
    return result;
}

}  // namespace double_ok_gesture
