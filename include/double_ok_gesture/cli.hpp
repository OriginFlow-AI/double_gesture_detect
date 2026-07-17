#pragma once

#include <string_view>

namespace double_ok_gesture {

// Strict CLI conversions: the complete token must match the requested type.
int parse_int_argument(std::string_view value, std::string_view option);
double parse_finite_double_argument(
    std::string_view value,
    std::string_view option);

}  // namespace double_ok_gesture
