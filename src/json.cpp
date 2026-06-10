#include "double_ok_gesture/json.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace double_ok_gesture {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Json parse() {
        skip_ws();
        Json value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            fail("trailing characters");
        }
        return value;
    }

private:
    Json parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) {
            fail("unexpected end of input");
        }
        const char ch = text_[pos_];
        if (ch == 'n') {
            expect("null");
            return Json(nullptr);
        }
        if (ch == 't') {
            expect("true");
            return Json(true);
        }
        if (ch == 'f') {
            expect("false");
            return Json(false);
        }
        if (ch == '"') {
            return Json(parse_string());
        }
        if (ch == '[') {
            return Json(parse_array());
        }
        if (ch == '{') {
            return Json(parse_object());
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            return Json(parse_number());
        }
        fail("unexpected character");
    }

    std::string parse_string() {
        consume('"');
        std::string result;
        while (pos_ < text_.size()) {
            const char ch = text_[pos_++];
            if (ch == '"') {
                return result;
            }
            if (ch == '\\') {
                if (pos_ >= text_.size()) {
                    fail("unterminated escape");
                }
                const char escaped = text_[pos_++];
                switch (escaped) {
                    case '"':
                    case '\\':
                    case '/':
                        result.push_back(escaped);
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u':
                        if (pos_ + 4 > text_.size()) {
                            fail("short unicode escape");
                        }
                        pos_ += 4;
                        result.push_back('?');
                        break;
                    default:
                        fail("unsupported escape");
                }
            } else {
                result.push_back(ch);
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        const std::size_t start = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        consume_digits();
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            consume_digits();
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            consume_digits();
        }
        return std::stod(text_.substr(start, pos_ - start));
    }

    Json::array_type parse_array() {
        consume('[');
        Json::array_type values;
        skip_ws();
        if (peek(']')) {
            ++pos_;
            return values;
        }
        while (true) {
            values.push_back(parse_value());
            skip_ws();
            if (peek(']')) {
                ++pos_;
                return values;
            }
            consume(',');
        }
    }

    Json::object_type parse_object() {
        consume('{');
        Json::object_type values;
        skip_ws();
        if (peek('}')) {
            ++pos_;
            return values;
        }
        while (true) {
            skip_ws();
            if (!peek('"')) {
                fail("object key must be a string");
            }
            std::string key = parse_string();
            skip_ws();
            consume(':');
            values.emplace(std::move(key), parse_value());
            skip_ws();
            if (peek('}')) {
                ++pos_;
                return values;
            }
            consume(',');
        }
    }

    void consume_digits() {
        const std::size_t start = pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        if (pos_ == start) {
            fail("expected digits");
        }
    }

    void expect(const std::string& token) {
        if (text_.compare(pos_, token.size(), token) != 0) {
            fail("expected " + token);
        }
        pos_ += token.size();
    }

    void consume(char expected) {
        skip_ws();
        if (!peek(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
        ++pos_;
    }

    bool peek(char expected) const {
        return pos_ < text_.size() && text_[pos_] == expected;
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON parse error at byte " + std::to_string(pos_) + ": " + message);
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

template <typename T>
const T& require_type(const Json::value_type& value, const std::string& name) {
    const T* typed = std::get_if<T>(&value);
    if (!typed) {
        throw std::runtime_error("JSON value is not a " + name);
    }
    return *typed;
}

}  // namespace

Json::Json(value_type value) : value_(std::move(value)) {}

bool Json::is_null() const {
    return std::holds_alternative<std::nullptr_t>(value_);
}

bool Json::is_bool() const {
    return std::holds_alternative<bool>(value_);
}

bool Json::is_number() const {
    return std::holds_alternative<double>(value_);
}

bool Json::is_string() const {
    return std::holds_alternative<std::string>(value_);
}

bool Json::is_array() const {
    return std::holds_alternative<array_type>(value_);
}

bool Json::is_object() const {
    return std::holds_alternative<object_type>(value_);
}

bool Json::as_bool() const {
    return require_type<bool>(value_, "bool");
}

double Json::as_number() const {
    return require_type<double>(value_, "number");
}

const std::string& Json::as_string() const {
    return require_type<std::string>(value_, "string");
}

const Json::array_type& Json::as_array() const {
    return require_type<array_type>(value_, "array");
}

const Json::object_type& Json::as_object() const {
    return require_type<object_type>(value_, "object");
}

const Json* Json::get(const std::string& key) const {
    if (!is_object()) {
        return nullptr;
    }
    const auto& object = as_object();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

Json parse_json(const std::string& text) {
    return Parser(text).parse();
}

Json load_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open JSON file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return parse_json(buffer.str());
}

}  // namespace double_ok_gesture
