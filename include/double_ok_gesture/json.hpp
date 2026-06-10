#pragma once

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace double_ok_gesture {

class Json {
public:
    using array_type = std::vector<Json>;
    using object_type = std::map<std::string, Json>;
    using value_type = std::variant<std::nullptr_t, bool, double, std::string, array_type, object_type>;

    Json() = default;
    explicit Json(value_type value);

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    bool as_bool() const;
    double as_number() const;
    const std::string& as_string() const;
    const array_type& as_array() const;
    const object_type& as_object() const;

    const Json* get(const std::string& key) const;

private:
    value_type value_ = nullptr;
};

Json parse_json(const std::string& text);
Json load_json(const std::filesystem::path& path);

}  // namespace double_ok_gesture
