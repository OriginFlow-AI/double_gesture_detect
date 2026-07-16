#include "double_ok_gesture/model_contract.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "double_ok_gesture/json.hpp"
#include "double_ok_gesture/yolov8_pose_postprocess.hpp"

namespace double_ok_gesture {
namespace {

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        for (std::size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    std::string finish() {
        bit_count_ += static_cast<std::uint64_t>(buffer_size_) * 8U;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) {
                buffer_[buffer_size_++] = 0;
            }
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) {
            buffer_[buffer_size_++] = 0;
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            buffer_[buffer_size_++] =
                static_cast<std::uint8_t>(bit_count_ >> shift);
        }
        transform(buffer_.data());

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t value : state_) {
            output << std::setw(8) << value;
        }
        return output.str();
    }

private:
    static std::uint32_t rotate_right(
        std::uint32_t value,
        std::uint32_t count) {
        return (value >> count) | (value << (32U - count));
    }

    void transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = index * 4;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24U) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t s0 =
                rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^
                (words[index - 15] >> 3U);
            const std::uint32_t s1 =
                rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^
                (words[index - 2] >> 10U);
            words[index] = words[index - 16] + s0 +
                           words[index - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sum1 =
                rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 =
                h + sum1 + choose + constants[index] + words[index];
            const std::uint32_t sum0 =
                rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            const std::uint32_t majority =
                (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t bit_count_ = 0;
};

const Json& require_member(const Json& object, const char* key) {
    const Json* value = object.get(key);
    if (!value) {
        throw std::runtime_error(
            std::string("manifest is missing '") + key + "'");
    }
    return *value;
}

std::uintmax_t manifest_uint(const Json& object, const char* key) {
    const double value = require_member(object, key).as_number();
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value ||
        value > static_cast<double>(
                    std::numeric_limits<std::uintmax_t>::max())) {
        throw std::runtime_error(
            std::string("manifest '") + key +
            "' must be a non-negative integer");
    }
    return static_cast<std::uintmax_t>(value);
}

void require_shape(
    const Json& value,
    const std::vector<std::uintmax_t>& expected,
    const std::string& name) {
    const auto& dimensions = value.as_array();
    if (dimensions.size() != expected.size()) {
        throw std::runtime_error(name + " has an unexpected rank");
    }
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        const double dimension = dimensions[index].as_number();
        if (!std::isfinite(dimension) || dimension < 0.0 ||
            std::floor(dimension) != dimension ||
            static_cast<std::uintmax_t>(dimension) != expected[index]) {
            throw std::runtime_error(
                name + " has an unexpected dimension");
        }
    }
}

void require_number_array(
    const Json& value,
    const std::vector<double>& expected,
    const std::string& name) {
    const auto& values = value.as_array();
    if (values.size() != expected.size()) {
        throw std::runtime_error(name + " has an unexpected length");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double actual = values[index].as_number();
        if (!std::isfinite(actual) ||
            std::abs(actual - expected[index]) > 1e-12) {
            throw std::runtime_error(name + " has an unexpected value");
        }
    }
}

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot open file for SHA-256: " + path.string());
    }
    Sha256 hash;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(),
                   static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash.update(
                reinterpret_cast<const std::uint8_t*>(buffer.data()),
                static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "failed while hashing file: " + path.string());
    }
    return hash.finish();
}

YoloV8PoseManifestInfo validate_yolov8_pose_model_contract(
    const std::filesystem::path& model_path,
    const std::filesystem::path& manifest_path,
    int expected_input_size) {
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error(
            "YOLOv8 RKNN model not found: " + model_path.string());
    }
    if (!std::filesystem::is_regular_file(manifest_path)) {
        throw std::runtime_error(
            "YOLOv8 model manifest not found: " + manifest_path.string());
    }
    if (expected_input_size < 32 || expected_input_size % 32 != 0) {
        throw std::invalid_argument(
            "YOLOv8 input size must be at least 32 and divisible by 32");
    }

    const Json root = load_json(manifest_path);
    (void)root.as_object();
    if (manifest_uint(root, "schema_version") != 1) {
        throw std::runtime_error("manifest schema_version must be 1");
    }

    YoloV8PoseManifestInfo info;
    info.model_kind = require_member(root, "model_kind").as_string();
    info.target_platform =
        require_member(root, "target_platform").as_string();
    info.dtype = require_member(root, "dtype").as_string();
    info.sha256 = require_member(root, "rknn_sha256").as_string();
    info.input_size = static_cast<int>(manifest_uint(root, "input_size"));
    if (info.model_kind != "yolov8n_pose_hand_21_rkopt") {
        throw std::runtime_error(
            "manifest model_kind is not yolov8n_pose_hand_21_rkopt");
    }
    if (info.target_platform != "rk3588") {
        throw std::runtime_error(
            "manifest target_platform must be rk3588");
    }
    if (info.dtype != "fp" && info.dtype != "i8") {
        throw std::runtime_error("manifest dtype must be fp or i8");
    }
    if (info.input_size != expected_input_size) {
        throw std::runtime_error(
            "configured YOLOv8 input size does not match manifest");
    }
    if (manifest_uint(root, "keypoint_count") != 21 ||
        manifest_uint(root, "keypoint_dimensions") != 3) {
        throw std::runtime_error(
            "manifest must declare 21 x,y,visibility keypoints");
    }
    const std::array<std::string, 3> dimension_order = {
        "x", "y", "visibility"};
    const auto& manifest_dimension_order =
        require_member(root, "keypoint_dimension_order").as_array();
    if (manifest_dimension_order.size() != dimension_order.size()) {
        throw std::runtime_error(
            "manifest keypoint_dimension_order must be x,y,visibility");
    }
    for (std::size_t index = 0; index < dimension_order.size(); ++index) {
        if (manifest_dimension_order[index].as_string() !=
            dimension_order[index]) {
            throw std::runtime_error(
                "manifest keypoint_dimension_order must be x,y,visibility");
        }
    }
    const std::array<std::string, 21> keypoint_order = {
        "wrist",
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    };
    const auto& manifest_keypoint_order =
        require_member(root, "keypoint_order").as_array();
    if (manifest_keypoint_order.size() != keypoint_order.size()) {
        throw std::runtime_error(
            "manifest keypoint_order must contain the 21 expected points");
    }
    for (std::size_t index = 0; index < keypoint_order.size(); ++index) {
        if (manifest_keypoint_order[index].as_string() !=
            keypoint_order[index]) {
            throw std::runtime_error(
                "manifest keypoint_order does not match runtime order");
        }
    }
    const auto candidates = static_cast<std::uintmax_t>(
        yolov8_pose::candidate_count_for_input(
            expected_input_size, expected_input_size));
    if (manifest_uint(root, "candidate_count") != candidates) {
        throw std::runtime_error(
            "manifest candidate_count does not match input size");
    }
    if (manifest_uint(root, "rknn_size") !=
        std::filesystem::file_size(model_path)) {
        throw std::runtime_error(
            "RKNN model size does not match manifest");
    }

    const Json& preprocess = require_member(root, "preprocess");
    if (require_member(preprocess, "host_input").as_string() !=
            "RGB uint8 NHWC" ||
        manifest_uint(preprocess, "letterbox_value") != 114) {
        throw std::runtime_error(
            "manifest preprocess must be RGB uint8 NHWC with letterbox 114");
    }
    require_number_array(
        require_member(preprocess, "mean"), {0.0, 0.0, 0.0},
        "manifest preprocess mean");
    require_number_array(
        require_member(preprocess, "std"), {255.0, 255.0, 255.0},
        "manifest preprocess std");
    const auto& outputs =
        require_member(require_member(root, "onnx"), "outputs").as_array();
    if (outputs.size() != 4) {
        throw std::runtime_error(
            "manifest ONNX contract must contain four outputs");
    }
    require_shape(
        require_member(outputs[0], "shape"),
        {1, 65,
         static_cast<std::uintmax_t>(expected_input_size / 8),
         static_cast<std::uintmax_t>(expected_input_size / 8)},
        "manifest output 0");
    require_shape(
        require_member(outputs[1], "shape"),
        {1, 65,
         static_cast<std::uintmax_t>(expected_input_size / 16),
         static_cast<std::uintmax_t>(expected_input_size / 16)},
        "manifest output 1");
    require_shape(
        require_member(outputs[2], "shape"),
        {1, 65,
         static_cast<std::uintmax_t>(expected_input_size / 32),
         static_cast<std::uintmax_t>(expected_input_size / 32)},
        "manifest output 2");
    require_shape(
        require_member(outputs[3], "shape"),
        {1, 21, 3, candidates},
        "manifest output 3");

    if (info.sha256.size() != 64 ||
        !std::all_of(
            info.sha256.begin(), info.sha256.end(),
            [](unsigned char ch) { return std::isxdigit(ch) != 0; })) {
        throw std::runtime_error(
            "manifest rknn_sha256 must contain 64 hexadecimal characters");
    }
    std::transform(
        info.sha256.begin(), info.sha256.end(), info.sha256.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    if (sha256_file(model_path) != info.sha256) {
        throw std::runtime_error(
            "RKNN model SHA-256 does not match manifest");
    }
    if (const Json* validation = root.get("board_validation")) {
        if (const Json* completed = validation->get("completed")) {
            info.board_validation_completed = completed->as_bool();
        }
    }
    return info;
}

}  // namespace double_ok_gesture
